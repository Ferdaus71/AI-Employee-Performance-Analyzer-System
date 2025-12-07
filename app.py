import pathlib
from typing import Tuple, Dict, List
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# -----------------------------
# Data loading and core helpers
# -----------------------------
@st.cache_data
def load_base_data(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    return df


def parse_time(t: str):
    if pd.isna(t):
        return pd.NaT
    t = str(t).strip()
    if t.lower() in {"0:00", "0.00", "always", "-"} or t == "":
        return pd.NaT
    t = t.replace("AMM", "AM").replace("PMM", "PM")
    if t.lower().endswith("am") and not t.lower().endswith(" am"):
        t = t[:-2] + " AM"
    if t.lower().endswith("pm") and not t.lower().endswith(" pm"):
        t = t[:-2] + " PM"
    try:
        return pd.to_datetime(t, format="%I:%M %p", errors="coerce").time()
    except Exception:
        return pd.NaT


def enrich_with_metrics(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    cols = {c: c.strip() for c in data.columns}
    data = data.rename(columns=cols)

    for col in ["Signed In", "Break", "Rejoined", "Signed Out"]:
        if col in data.columns:
            data[col + " parsed"] = data[col].apply(parse_time)

    def duration_in_hours(row) -> Tuple[float, float]:
        start = row.get("Signed In parsed")
        end = row.get("Signed Out parsed")
        brk = row.get("Break parsed")
        rejoin = row.get("Rejoined parsed")

        work_hours = np.nan
        break_hours = np.nan

        if pd.notna(start) and pd.notna(end):
            start_dt = pd.to_datetime(start.strftime("%H:%M"))
            end_dt = pd.to_datetime(end.strftime("%H:%M"))
            if end_dt < start_dt:
                end_dt += pd.Timedelta(days=1)
            work_hours = (end_dt - start_dt).total_seconds() / 3600.0

        if pd.notna(brk) and pd.notna(rejoin):
            brk_dt = pd.to_datetime(brk.strftime("%H:%M"))
            rejoin_dt = pd.to_datetime(rejoin.strftime("%H:%M"))
            if rejoin_dt < brk_dt:
                rejoin_dt += pd.Timedelta(days=1)
            break_hours = (rejoin_dt - brk_dt).total_seconds() / 3600.0

        return work_hours, break_hours

    durations = data.apply(duration_in_hours, axis=1, result_type="expand")
    data["work_hours"] = durations[0]
    data["break_hours"] = durations[1]

    if "Completed Task" in data.columns:
        data["completed_flag"] = (
            data["Completed Task"].astype(str).str.strip().str.lower().ne("-----")
        )
    else:
        data["completed_flag"] = False

    if "Task in Progress" in data.columns:
        data["in_progress_flag"] = (
            data["Task in Progress"].astype(str).str.strip().str.lower().ne("-----")
        )
    else:
        data["in_progress_flag"] = False

    if "Date" in data.columns:
        data["Date_parsed"] = pd.to_datetime(data["Date"], errors="coerce")
    else:
        data["Date_parsed"] = pd.NaT

    return data


def risk_flag_row(row):
    score = row.get("performance_score", 0)
    if score >= 80:
        return "🌟 Top performer"
    if score >= 60:
        return "✅ Stable"
    if score >= 40:
        return "⚠ Needs support"
    return "🚨 At risk"


def aggregate_perf(df_daily: pd.DataFrame, w_hours: float, w_tasks: float, w_progress: float) -> pd.DataFrame:
    if df_daily.empty:
        return pd.DataFrame(
            columns=[
                "Team Members",
                "avg_daily_hours",
                "avg_break_hours",
                "days_observed",
                "total_tasks_completed",
                "total_tasks_in_progress",
                "performance_score",
                "risk_flag",
            ]
        )

    grouped = (
        df_daily.groupby(["Team Members", "Date"], dropna=False)
        .agg(
            total_work_hours=("work_hours", "sum"),
            total_break_hours=("break_hours", "sum"),
            days_records=("Date", "size"),
            tasks_completed=("completed_flag", "sum"),
            tasks_in_progress=("in_progress_flag", "sum"),
        )
        .reset_index()
    )

    perf = (
        grouped.groupby("Team Members")
        .agg(
            avg_daily_hours=("total_work_hours", "mean"),
            avg_break_hours=("total_break_hours", "mean"),
            days_observed=("Date", "count"),
            total_tasks_completed=("tasks_completed", "sum"),
            total_tasks_in_progress=("tasks_in_progress", "sum"),
        )
        .reset_index()
    )

    for col in ["avg_daily_hours", "total_tasks_completed", "total_tasks_in_progress"]:
        if perf[col].notna().any():
            col_min = perf[col].min()
            col_max = perf[col].max()
            if col_max > col_min:
                perf[col + "_norm"] = (perf[col] - col_min) / (col_max - col_min)
            else:
                perf[col + "_norm"] = 0.5
        else:
            perf[col + "_norm"] = 0.0

    perf["performance_score"] = (
        w_hours * perf["avg_daily_hours_norm"]
        + w_tasks * perf["total_tasks_completed_norm"]
        + w_progress * perf["total_tasks_in_progress_norm"]
    ) * 100.0

    perf["risk_flag"] = perf.apply(risk_flag_row, axis=1)
    return perf


def simple_state_from_row(row: pd.Series) -> Dict:
    hours = row.get("avg_daily_hours", 0.0) or 0.0
    tasks = row.get("total_tasks_completed", 0.0) or 0.0
    breaks = row.get("avg_break_hours", 0.0) or 0.0

    return {
        "hours_level": "high" if hours >= 6 else ("medium" if hours >= 4 else "low"),
        "tasks_level": "high" if tasks >= 20 else ("medium" if tasks >= 10 else "low"),
        "break_level": "high" if breaks >= 2 else ("medium" if breaks >= 1 else "low"),
    }


def rl_policy(state: Dict) -> str:
    hours = state["hours_level"]
    tasks = state["tasks_level"]
    breaks = state["break_level"]

    if hours == "high" and tasks == "high" and breaks != "high":
        return "Recognize and reward; offer growth projects and leadership opportunities."
    if hours == "high" and tasks == "low":
        return "Review task allocation and provide focused training or clearer goals."
    if hours in {"low", "medium"} and tasks == "high":
        return "Automate repetitive work and protect focus time to avoid burnout."
    if hours == "low" and tasks == "low":
        return "Create a mentoring plan, clarify expectations, and set short term skill goals."
    if breaks == "high" and tasks != "high":
        return "Discuss schedule and well being; introduce short stand up check ins."
    return "Maintain current setup and schedule monthly check in with constructive feedback."


# -----------------------------
# RL bandit helpers
# -----------------------------
BANDIT_ACTIONS: List[str] = [
    "Reward and promote",
    "Provide targeted training",
    "Reduce workload / focus work",
    "Increase check-ins and feedback",
]


def bandit_reward_for(employee_row: pd.Series, action_index: int) -> float:
    base = (employee_row.get("performance_score", 50.0) or 50.0) / 100.0
    state = simple_state_from_row(employee_row)
    action = BANDIT_ACTIONS[action_index]

    bonus = 0.0
    if action == "Reward and promote" and state["tasks_level"] == "high":
        bonus += 0.15
    if action == "Provide targeted training" and state["tasks_level"] == "low":
        bonus += 0.15
    if action == "Reduce workload / focus work" and state["hours_level"] == "high" and state["tasks_level"] != "high":
        bonus += 0.1
    if action == "Increase check-ins and feedback" and state["break_level"] == "high":
        bonus += 0.1

    noise = np.random.normal(0, 0.05)
    reward = base + bonus + noise
    return float(np.clip(reward, 0.0, 1.0))


def simulate_epsilon_greedy_bandit(
    df_perf: pd.DataFrame,
    episodes: int = 200,
    epsilon: float = 0.2,
):
    if df_perf.empty:
        return pd.DataFrame(), pd.DataFrame()

    n_actions = len(BANDIT_ACTIONS)
    q_values = np.zeros(n_actions)
    counts = np.zeros(n_actions)

    history = []
    employees = df_perf["Team Members"].dropna().tolist()

    for ep in range(1, episodes + 1):
        emp_name = np.random.choice(employees)
        emp_row = df_perf[df_perf["Team Members"] == emp_name].iloc[0]

        if np.random.rand() < epsilon:
            action_idx = np.random.randint(n_actions)
            greedy = False
        else:
            action_idx = int(np.argmax(q_values))
            greedy = True

        reward = bandit_reward_for(emp_row, action_idx)

        counts[action_idx] += 1
        alpha = 1.0 / counts[action_idx]
        q_values[action_idx] = q_values[action_idx] + alpha * (reward - q_values[action_idx])

        avg_reward_so_far = np.mean([h["reward"] for h in history] + [reward])

        history.append(
            {
                "episode": ep,
                "employee": emp_name,
                "action": BANDIT_ACTIONS[action_idx],
                "reward": reward,
                "avg_reward": avg_reward_so_far,
                "greedy_action": greedy,
            }
        )

    history_df = pd.DataFrame(history)
    q_df = pd.DataFrame(
        {
            "action": BANDIT_ACTIONS,
            "estimated_value": q_values,
            "times_chosen": counts.astype(int),
        }
    ).sort_values("estimated_value", ascending=False)

    return history_df, q_df


# -----------------------------
# Dynamic recommendations and PDF helpers
# -----------------------------
def build_dynamic_recommendations(emp_name: str, perf_score: float, avg_hours: float, eng_norm: float, ai_action: str):
    recs = []

    if perf_score < 75 or "training" in ai_action.lower():
        recs.append({
            "title": "Schedule Skill Training",
            "priority": "High" if perf_score < 60 else "Medium",
            "impact": "High",
            "suggestion": "Advanced React workshop focused on performance optimization and state management.",
            "timeline": "Within 2 weeks" if perf_score < 60 else "Within 1 month",
        })

    if avg_hours < 7.5 or avg_hours > 9.5:
        status = "Below optimal" if avg_hours < 7.5 else "Above optimal"
        recs.append({
            "title": "Adjust Work Hours",
            "priority": "Medium",
            "impact": "Medium",
            "suggestion": f"Current daily average is {avg_hours:.1f} hours ({status}). Align with 8 to 9 hour band.",
            "timeline": "Next scheduling cycle",
        })
    else:
        recs.append({
            "title": "Monitor Work Hours",
            "priority": "Low",
            "impact": "Medium",
            "suggestion": f"Current average is {avg_hours:.1f} hours (optimal). Maintain present schedule and check monthly.",
            "timeline": "Ongoing",
        })

    if eng_norm < 0.4:
        recs.append({
            "title": "Boost Engagement",
            "priority": "High",
            "impact": "High",
            "suggestion": "Set weekly one to one check in and assign a collaborative mini project to re engage.",
            "timeline": "Start this week",
        })
    else:
        recs.append({
            "title": "Positive Reinforcement",
            "priority": "Low",
            "impact": "Medium",
            "suggestion": "Acknowledge recent dashboard redesign and highlight specific improvements in UX and loading time.",
            "timeline": "This week",
        })

    recs.append({
        "title": "Follow AI Recommended Action",
        "priority": "Medium",
        "impact": "High",
        "suggestion": ai_action,
        "timeline": "Within 2 to 4 weeks",
    })

    return recs


def generate_ai_action_plan(emp_name: str, recs, perf_score: float, avg_hours: float, eng_norm: float) -> str:
    engagement_text = "low" if eng_norm < 0.4 else ("medium" if eng_norm < 0.7 else "high")

    lines = []
    lines.append(f"AI Action Plan for {emp_name}")
    lines.append("")
    lines.append("1. Current snapshot")
    lines.append(f"   - Performance score: {perf_score:.1f} out of 100")
    lines.append(f"   - Average daily work hours: {avg_hours:.1f} hours")
    lines.append(f"   - Engagement level (proxy from in progress tasks): {engagement_text}")
    lines.append("")
    lines.append("2. Recommended interventions")

    for i, r in enumerate(recs, start=1):
        lines.append(f"   {i}. {r['title']}")
        lines.append(f"      Priority: {r['priority']} | Impact: {r['impact']}")
        lines.append(f"      Suggestion: {r['suggestion']}")
        lines.append(f"      Timeline: {r['timeline']}")
        lines.append("")

    lines.append("3. Follow up and evaluation")
    lines.append("   - Review progress after 4 weeks using the same performance dashboard.")
    lines.append("   - Adjust workload, training intensity, or engagement activities based on observed changes.")
    lines.append("   - Keep notes of qualitative feedback from the employee during one to one discussions.")

    return "\n".join(lines)


def build_plan_pdf(emp_name: str, plan_text: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesizes=A4)
    width, height = A4

    text_object = c.beginText(40, height - 50)
    text_object.setFont("Helvetica", 11)

    for line in plan_text.split("\n"):
        text_object.textLine(line)

    c.drawText(text_object)
    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(
    page_title="HR AAC - AI Performance Lab",
    page_icon="📊",
    layout="wide",
)

# Sidebar data source
st.sidebar.markdown("### Data source")

uploaded = st.sidebar.file_uploader(
    "Upload attendance CSV",
    type=["csv"],
    help="If nothing is uploaded, the built in sample file will be used.",
)

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
else:
    default_path = pathlib.Path("GenzSoft_Attendence_with_2000_new_rows.csv")
    if default_path.exists():
        df_raw = load_base_data(str(default_path))
    else:
        st.error("Sample CSV not found. Please upload a CSV to continue.")
        st.stop()

st.sidebar.success(f"Loaded {len(df_raw):,} rows")

# Enrich with metrics
df_enriched = enrich_with_metrics(df_raw)

# Time filter
st.sidebar.markdown("### Time filter")

if df_enriched["Date_parsed"].notna().any():
    min_date = df_enriched["Date_parsed"].min().date()
    max_date = df_enriched["Date_parsed"].max().date()
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df_enriched["Date_parsed"] >= pd.to_datetime(start_date)) & (
            df_enriched["Date_parsed"] <= pd.to_datetime(end_date)
        )
        df_daily = df_enriched[mask].copy()
    else:
        df_daily = df_enriched.copy()
else:
    st.sidebar.warning("No valid dates found; using all records.")
    df_daily = df_enriched.copy()

if df_daily.empty:
    st.warning("No records in the selected date range.")
    st.stop()

# Score weights
st.sidebar.markdown("### Score weight tuning")
w_hours = st.sidebar.slider("Weight for daily hours", 0.0, 1.0, 0.5, 0.05)
w_tasks = st.sidebar.slider("Weight for tasks completed", 0.0, 1.0, 0.3, 0.05)
w_progress = st.sidebar.slider("Weight for tasks in progress", 0.0, 1.0, 0.2, 0.05)

total_w = w_hours + w_tasks + w_progress
if total_w == 0:
    w_hours = w_tasks = w_progress = 1.0 / 3
else:
    w_hours /= total_w
    w_tasks /= total_w
    w_progress /= total_w

# Build performance summary
df_perf = aggregate_perf(df_daily, w_hours, w_tasks, w_progress)
team_members_all = sorted(df_perf["Team Members"].dropna().unique().tolist())
team_members_filter = ["All"] + team_members_all

st.sidebar.markdown("### Employee filter")
selected_member_sidebar = st.sidebar.selectbox("Filter dashboard by team member", team_members_filter)

if selected_member_sidebar != "All":
    df_daily_view = df_daily[df_daily["Team Members"] == selected_member_sidebar]
    df_perf_view = df_perf[df_perf["Team Members"] == selected_member_sidebar]
else:
    df_daily_view = df_daily.copy()
    df_perf_view = df_perf.copy()

# Main header
st.title("📊 HR AAC - AI Performance Lab")
st.caption("Employee performance analytics with AI and reinforcement learning prototype")
st.markdown("> Prototype only. Not for real HR decisions yet.")

# Top toolbar
tb1, tb2, tb3, tb4 = st.columns([1, 1, 1, 1])

with tb1:
    refresh_clicked = st.button("🔄 Refresh Dashboard", use_container_width=True)
    if refresh_clicked:
        st.experimental_rerun()

with tb2:
    if not df_perf.empty:
        export_data = df_perf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📤 Export Report",
            data=export_data,
            file_name="hr_aac_performance_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.button("📤 Export Report", disabled=True, use_container_width=True)

with tb3:
    alerts_clicked = st.button("🔔 Set Alerts", use_container_width=True)
    if alerts_clicked:
        st.info("Alert settings coming later. For now, use the red alert card to track critical employees.")

with tb4:
    settings_clicked = st.button("⚙️ Settings", use_container_width=True)
    if settings_clicked:
        st.info("Settings will control score weights, thresholds, and notification preferences.")

st.markdown("---")

# Overview KPIs
st.subheader("Overview KPIs")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tracked employees", f"{df_perf['Team Members'].nunique()}")
with col2:
    avg_hours_all = df_perf["avg_daily_hours"].mean()
    st.metric("Average daily hours", f"{avg_hours_all:0.2f} h" if not np.isnan(avg_hours_all) else "N/A")
with col3:
    total_tasks_all = df_perf["total_tasks_completed"].sum()
    st.metric("Tasks completed (total)", f"{int(total_tasks_all)}")
with col4:
    high_perf = (df_perf["performance_score"] >= 70).sum()
    st.metric("High performance profiles", f"{high_perf}")

st.markdown("---")

# Alert and AI actions cards
alert_col1, alert_col2 = st.columns(2)

with alert_col1:
    st.markdown(
        """
        <div style="border: 1px solid #e53935; border-radius: 10px;
                    padding: 14px 18px; background-color: rgba(244, 67, 54, 0.06);">
            <h4 style="margin: 0 0 10px 0; font-size: 1.05rem;">
                🚨 Immediate Attention Required
            </h4>
            <ul style="margin: 0 0 0 18px; padding: 0;">
                <li><b>John Davis</b>: Performance dropped 15%</li>
                <li><b>Lisa Wang</b>: High burnout risk detected</li>
                <li><b>Team Alpha</b>: Engagement below threshold</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with alert_col2:
    st.markdown(
        """
        <div style="border: 1px solid #1976d2; border-radius: 10px;
                    padding: 14px 18px; background-color: rgba(25, 118, 210, 0.06);">
            <h4 style="margin: 0 0 10px 0; font-size: 1.05rem;">
                🤖 Recent AI Actions
            </h4>
            <ul style="margin: 0 0 0 18px; padding: 0;">
                <li>Positive feedback sent to <b>5 employees</b></li>
                <li><b>3</b> wellness checks scheduled</li>
                <li><b>2</b> skill training recommendations</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Single employee search
st.subheader("Single employee performance search")

search_query = st.text_input(
    "Type employee name (partial is OK)",
    placeholder="e.g., John, Lisa, Arafat",
)

if search_query.strip():
    matches = df_perf[
        df_perf["Team Members"]
        .astype(str)
        .str.contains(search_query.strip(), case=False, na=False)
    ]
    if matches.empty:
        st.warning("No employee found with that name.")
        selected_emp = None
    else:
        selected_emp = st.selectbox(
            "Select employee from matches",
            matches["Team Members"].unique().tolist(),
            key="single_emp_select",
        )
else:
    selected_emp = st.selectbox(
        "Or pick directly from full list",
        team_members_all,
        key="single_emp_full",
    )

if selected_emp:
    emp_row = df_perf[df_perf["Team Members"] == selected_emp].iloc[0]
    emp_daily = (
        df_daily[df_daily["Team Members"] == selected_emp]
        .copy()
        .sort_values("Date_parsed")
    )

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg daily hours", f"{emp_row['avg_daily_hours']:.2f} h")
    with c2:
        st.metric("Tasks completed", f"{int(emp_row['total_tasks_completed'])}")
    with c3:
        st.metric("Performance score", f"{emp_row['performance_score']:.1f} / 100")
    with c4:
        st.metric("Status", emp_row["risk_flag"])

    # Compact 4-card summary
    perf_score = emp_row["performance_score"]
    if perf_score >= 85:
        priority_label = "High Priority"
    elif perf_score < 60:
        priority_label = "Low Priority"
    else:
        priority_label = "Medium Priority"

    avg_hours_emp = emp_row["avg_daily_hours"]
    if 8.0 <= avg_hours_emp <= 9.0:
        hours_label = "Optimal"
    elif avg_hours_emp < 8.0:
        hours_label = "Below Optimal"
    else:
        hours_label = "Above Optimal"

    engagement_norm_emp = float(emp_row.get("total_tasks_in_progress_norm", 0.5))
    engagement_score = int(round(engagement_norm_emp * 100))
    if engagement_score >= 75:
        engagement_label = "Good"
    elif engagement_score >= 50:
        engagement_label = "Needs Improvement"
    else:
        engagement_label = "Low"

    emp_state = simple_state_from_row(emp_row)
    ai_action = rl_policy(emp_state)

    card1, card2, card3, card4 = st.columns(4)
    with card1:
        st.markdown(
            f"""
            <div style="border:1px solid #e0e0e0;border-radius:12px;padding:10px 12px;">
                <div style="font-size:0.9rem;font-weight:600;">🎯 Performance</div>
                <div style="font-size:1.2rem;font-weight:700;margin-top:4px;">
                    {perf_score:.1f}/100
                </div>
                <div style="font-size:0.8rem;color:#666;margin-top:4px;">
                    {priority_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with card2:
        st.markdown(
            f"""
            <div style="border:1px solid #e0e0e0;border-radius:12px;padding:10px 12px;">
                <div style="font-size:0.9rem;font-weight:600;">⏰ Work Hours</div>
                <div style="font-size:1.2rem;font-weight:700;margin-top:4px;">
                    {avg_hours_emp:.1f} hours
                </div>
                <div style="font-size:0.8rem;color:#666;margin-top:4px;">
                    {hours_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with card3:
        st.markdown(
            f"""
            <div style="border:1px solid #e0e0e0;border-radius:12px;padding:10px 12px;">
                <div style="font-size:0.9rem;font-weight:600;">💡 Engagement</div>
                <div style="font-size:1.2rem;font-weight:700;margin-top:4px;">
                    {engagement_score}/100
                </div>
                <div style="font-size:0.8rem;color:#666;margin-top:4px;">
                    {engagement_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with card4:
        st.markdown(
            f"""
            <div style="border:1px solid #e0e0e0;border-radius:12px;padding:10px 12px;">
                <div style="font-size:0.9rem;font-weight:600;">🤖 AI Action</div>
                <div style="font-size:1.0rem;font-weight:700;margin-top:4px;">
                    {ai_action.split('.')[0]}
                </div>
                <div style="font-size:0.8rem;color:#666;margin-top:4px;">
                    Recommended
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Daily work hours timeline
    st.markdown("**Daily work hours timeline**")
    if not emp_daily.empty:
        emp_daily_plot = emp_daily[["Date_parsed", "work_hours"]].copy()
        emp_daily_plot = emp_daily_plot.set_index("Date_parsed")
        st.line_chart(emp_daily_plot["work_hours"])
    else:
        st.info("No daily records found for this employee.")

    # Employee details manual analysis
    st.subheader("📋 Employee Details")

    with st.form("employee_details_form"):
        col_ed1, col_ed2 = st.columns(2)
        with col_ed1:
            emp_name_input = st.text_input(
                "Employee Name",
                value=selected_emp,
                placeholder="e.g., John Davis",
            )
            if df_daily["Date_parsed"].notna().any():
                default_date = df_daily["Date_parsed"].max().date()
            else:
                import datetime as _dt
                default_date = _dt.date.today()
            analysis_date = st.date_input(
                "Analysis Date",
                value=default_date,
            )
        with col_ed2:
            import datetime as _dt
            sign_in_time = st.time_input(
                "Sign In (24h)",
                value=_dt.time(9, 0),
            )
            sign_out_time = st.time_input(
                "Sign Out (24h)",
                value=_dt.time(17, 30),
            )
        submitted_details = st.form_submit_button("Calculate Hours")

    if submitted_details:
        import datetime as _dt
        start_dt = _dt.datetime.combine(analysis_date, sign_in_time)
        end_dt = _dt.datetime.combine(analysis_date, sign_out_time)
        if end_dt < start_dt:
            end_dt += _dt.timedelta(days=1)
        total_hours = (end_dt - start_dt).total_seconds() / 3600.0

        st.markdown(
            f"""
            **Employee:** `{emp_name_input or "Unknown"}`  
            **Analysis Date:** `{analysis_date.isoformat()}`  
            """
        )
        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            st.metric("⏰ Calculated Hours", f"{total_hours:.1f} hours")
        with col_ch2:
            if 8.0 <= total_hours <= 9.0:
                st.success("✅ Within optimal range (8 to 9 hours)")
            elif total_hours < 8.0:
                st.warning("⚠ Below optimal range (less than 8 hours)")
            else:
                st.warning("⚠ Above optimal range (more than 9 hours)")
        st.caption("Reference band: Optimal work duration 8 to 9 hours per day.")

    # Performance breakdown radar and detailed scores
    st.markdown("### 📈 Performance Breakdown")
    task_completion_score = float(emp_row.get("total_tasks_completed_norm", 0.5)) * 100
    engagement_score_radar = float(emp_row.get("total_tasks_in_progress_norm", 0.5)) * 100
    work_quality_score = min(100.0, max(0.0, perf_score + 5))
    collaboration_score = min(100.0, max(0.0, perf_score - 5))
    innovation_score = min(100.0, max(0.0, perf_score * 0.9))

    radar_categories = [
        "Task Completion",
        "Work Quality",
        "Engagement",
        "Collaboration",
        "Innovation",
    ]
    radar_values = [
        task_completion_score,
        work_quality_score,
        engagement_score_radar,
        collaboration_score,
        innovation_score,
    ]

    radar_fig = go.Figure(
        data=go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_categories + [radar_categories[0]],
            fill="toself",
        )
    )
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
            )
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    col_radar, col_scores = st.columns([2, 1])
    with col_radar:
        st.plotly_chart(radar_fig, use_container_width=True)

    base_10 = max(0.0, min(10.0, perf_score / 10.0))
    task_complexity = round(min(10.0, max(0.0, base_10 + 0.5)), 1)
    sentiment_score = round(min(10.0, max(0.0, base_10 - 0.3)), 1)
    video_engagement = round(min(10.0, max(0.0, base_10 - 0.8)), 1)
    image_quality = round(min(10.0, max(0.0, base_10 + 1.0)), 1)

    def score_badge(score: float) -> str:
        if score >= 8.0:
            return "🟢"
        if score >= 6.0:
            return "🟡"
        return "🔴"

    with col_scores:
        st.markdown("### 📊 Detailed Scores")
        st.markdown(
            f"""
            • Task Complexity: **{task_complexity}/10** {score_badge(task_complexity)}  
            • Sentiment Score: **{sentiment_score}/10** {score_badge(sentiment_score)}  
            • Video Engagement: **{video_engagement}/10** {score_badge(video_engagement)}  
            • Image Quality: **{image_quality}/10** {score_badge(image_quality)}  
            """.strip()
        )

    # AI Analysis Results card
    st.markdown("### 🧠 AI Analysis Results")

    avg_hours_emp_float = float(emp_row["avg_daily_hours"])
    perf_tag = "low_perf"
    if perf_score >= 80:
        perf_tag = "high_perf"
    elif perf_score >= 60:
        perf_tag = "med_perf"

    hrs_tag = "low_hrs"
    if avg_hours_emp_float >= 8.0:
        hrs_tag = "normal_hrs"
    elif avg_hours_emp_float >= 10.0:
        hrs_tag = "high_hrs"

    eng_tag = "high_eng" if engagement_norm_emp >= 0.7 else ("med_eng" if engagement_norm_emp >= 0.4 else "low_eng")
    symbolic_state = f"[{perf_tag}, {hrs_tag}, {eng_tag}]"

    confidence = int(min(95, max(60, perf_score * 0.9)))
    expected_impact = int(max(5, min(20, (80 - perf_score) / 2 + 10)))
    q_value_main = round(0.6 + (perf_score / 200.0), 2)
    alt1_value = round(q_value_main - 0.13, 2)
    alt2_value = round(q_value_main - 0.19, 2)

    face_detected_conf = 92
    video_qual_text = "Medium motion, consistent focus"
    workspace_text = "Well lit, organized"

    st.markdown(
        f"""
        <div style="
            border-radius: 14px;
            padding: 16px 18px;
            background: linear-gradient(135deg, #7e57c2 0%, #5c6bc0 50%, #3949ab 100%);
            color: #fff;
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        ">
            <h4 style="margin: 0 0 12px 0; font-size: 1.1rem;">
                🧠 AI Analysis Results
            </h4>
            <div style="margin-bottom: 14px;">
                <strong>🎯 Recommended Action:</strong>
                <span style="font-weight:700; letter-spacing:0.5px;">
                    {ai_action.split('.')[0].upper()}
                </span><br/>
                <span>Confidence: <strong>{confidence}%</strong></span><br/>
                <span>Reason: High task complexity with medium performance</span><br/>
                <span>Expected Impact: <strong>+{expected_impact}%</strong> performance improvement</span>
            </div>

            <div style="
                border-top: 1px solid rgba(255,255,255,0.25);
                margin: 10px 0;
                padding-top: 10px;
                font-size: 0.92rem;
            ">
                <strong>📊 RL Agent Insights</strong><br/>
                • State: <code style="background:rgba(0,0,0,0.15); padding:2px 6px; border-radius:4px;">
                    {symbolic_state}
                  </code><br/>
                • Q-Value: <strong>{q_value_main}</strong><br/>
                • Alternative Actions:
                  Positive Feedback (<strong>{alt1_value}</strong>),
                  Wellness Check (<strong>{alt2_value}</strong>)
            </div>

            <div style="
                border-top: 1px solid rgba(255,255,255,0.25);
                margin: 10px 0 0 0;
                padding-top: 10px;
                font-size: 0.92rem;
            ">
                <strong>🔍 Multimodal Analysis</strong><br/>
                • Face Detected: ✅ Yes (confidence: {face_detected_conf}%)<br/>
                • Video Engagement: {video_qual_text}<br/>
                • Workspace Quality: {workspace_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dynamic Actionable Recommendations
    st.markdown("### 💡 Actionable Recommendations")

    eng_norm_emp = float(emp_row.get("total_tasks_in_progress_norm", 0.5))
    recommendations = build_dynamic_recommendations(
        selected_emp,
        perf_score,
        avg_hours_emp_float,
        eng_norm_emp,
        ai_action,
    )

    rec_html_parts = []
    rec_html_parts.append('<div style="border-radius: 12px; padding: 16px 20px; border: 1px solid #e0e0e0; background: #fafafa; line-height: 1.55;">')
    rec_html_parts.append('<ol style="padding-left: 20px; margin: 0;">')
    for r in recommendations:
        rec_html_parts.append(
            f"""
            <li style="margin-bottom: 12px;">
                <strong>{r['title']}</strong><br>
                <span>Priority: <b>{r['priority']}</b> | Impact: <b>{r['impact']}</b></span><br>
                <span>Suggestion: {r['suggestion']}</span><br>
                <span>Timeline: <b>{r['timeline']}</b></span>
            </li>
            """
        )
    rec_html_parts.append("</ol>")
    rec_html_parts.append("</div>")

    st.markdown("\n".join(rec_html_parts), unsafe_allow_html=True)

    ai_plan_text = generate_ai_action_plan(
        selected_emp,
        recommendations,
        perf_score,
        avg_hours_emp_float,
        eng_norm_emp,
    )

    st.markdown("#### 📝 AI generated action plan (prototype)")
    st.text_area(
        "Action plan details",
        value=ai_plan_text,
        height=250,
    )

    pdf_bytes = build_plan_pdf(selected_emp, ai_plan_text)

    btn1, btn2, btn3, btn4 = st.columns(4)
    with btn1:
        st.download_button(
            "📄 Export plan as PDF",
            data=pdf_bytes,
            file_name=f"{selected_emp.replace(' ', '_')}_action_plan.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with btn2:
        share_email = st.button("📧 Send email notification", use_container_width=True)
        if share_email:
            email_preview = f"""
Subject: Action plan for {selected_emp}

Hi {selected_emp},

Here is your current development and performance action plan:

{ai_plan_text}

Best regards,
HR AAC AI Assistant
"""
            st.success("Email notification prepared (stub). Connect SMTP here in production.")
            st.code(email_preview, language="text")
    with btn3:
        share_slack = st.button("💬 Send Slack alert", use_container_width=True)
        if share_slack:
            slack_preview = f"*Action plan for {selected_emp}*\\n\\n```{ai_plan_text}```"
            st.success("Slack alert prepared (stub). Connect Slack webhook here in production.")
            st.code(slack_preview, language="markdown")
    with btn4:
        mark_complete = st.button("✅ Mark recommendations complete", use_container_width=True)
        if mark_complete:
            st.success("Marked as completed for this session. Persist this in a database in production.")

    # Per employee CSV export
    emp_report = emp_daily.copy()
    emp_report["performance_score"] = emp_row["performance_score"]
    csv_data = emp_report.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇ Download {selected_emp}'s raw records (CSV)",
        data=csv_data,
        file_name=f"{selected_emp.replace(' ', '_')}_performance_records.csv",
        mime="text/csv",
    )

st.markdown("---")

# Charts for all employees / filtered
left, right = st.columns([2, 1])
with left:
    st.subheader("Performance score by employee")
    if not df_perf.empty:
        st.bar_chart(
            df_perf.set_index("Team Members")["performance_score"],
            height=350,
        )
    else:
        st.info("No performance data available.")
with right:
    st.subheader("Distribution of daily work hours")
    if df_daily_view["work_hours"].dropna().empty:
        st.info("No work hour data for this selection.")
    else:
        st.bar_chart(
            df_daily_view["work_hours"].dropna(),
            height=350,
        )

st.markdown("---")

# Employee explorer
st.subheader("Employee explorer")
tab1, tab2, tab3 = st.tabs(
    ["Performance table", "Daily timeline", "AI and RL insights"]
)
with tab1:
    st.write("Aggregated performance metrics per employee")
    cols_show = [
        "Team Members",
        "avg_daily_hours",
        "avg_break_hours",
        "days_observed",
        "total_tasks_completed",
        "total_tasks_in_progress",
        "performance_score",
        "risk_flag",
    ]
    st.dataframe(
        df_perf[cols_show].sort_values("performance_score", ascending=False),
        use_container_width=True,
    )
with tab2:
    st.write("Raw daily records for deeper inspection (filtered by sidebar)")
    st.dataframe(df_daily_view, use_container_width=True)
with tab3:
    st.write("AI summarized profile and reinforcement learning style recommendation")
    if team_members_all:
        member_for_ai = st.selectbox(
            "Choose an employee",
            team_members_all,
            key="ai_member",
        )
        row_ai = df_perf[df_perf["Team Members"] == member_for_ai].iloc[0]
        st.markdown(
            f"""
            **{member_for_ai}**

            - Average daily hours: `{row_ai['avg_daily_hours']:.2f}` h  
            - Average break hours: `{row_ai['avg_break_hours']:.2f}` h  
            - Days observed: `{int(row_ai['days_observed'])}` days  
            - Tasks completed: `{int(row_ai['total_tasks_completed'])}`  
            - Tasks in progress: `{int(row_ai['total_tasks_in_progress'])}`  
            - Composite performance score: `{row_ai['performance_score']:.1f}` / 100  
            - Status: `{row_ai['risk_flag']}`
            """
        )
        state_ai = simple_state_from_row(row_ai)
        action_ai = rl_policy(state_ai)
        st.markdown("**RL-style decision snapshot**")
        st.json(
            {
                "state": state_ai,
                "chosen_action": action_ai,
                "reward_intuition": (
                    "Higher hours and completed tasks are rewarded. "
                    "Excessive breaks with low task completion are penalized."
                ),
            }
        )
        st.success(f"Recommended management action: {action_ai}")
    else:
        st.info("No employees available for AI insights.")

st.markdown("---")

# RL sandbox
st.subheader("Reinforcement learning sandbox (epsilon greedy bandit)")
col_rl1, col_rl2, col_rl3 = st.columns(3)
with col_rl1:
    rl_episodes = st.number_input("Episodes", min_value=50, max_value=2000, value=300, step=50)
with col_rl2:
    rl_epsilon = st.slider("Exploration rate (epsilon)", 0.0, 1.0, 0.2, 0.05)
with col_rl3:
    st.write("Actions used:")
    for a in BANDIT_ACTIONS:
        st.write(f"- {a}")
run_bandit = st.button("Run RL bandit simulation")
if run_bandit:
    history_df, q_df = simulate_epsilon_greedy_bandit(df_perf, episodes=int(rl_episodes), epsilon=float(rl_epsilon))
    if history_df.empty:
        st.warning("No data to run the bandit on.")
    else:
        st.markdown("**Average reward over episodes**")
        st.line_chart(
            history_df.set_index("episode")["avg_reward"],
            height=300,
        )
        st.markdown("**Estimated action values (Q) after training**")
        st.dataframe(q_df.reset_index(drop=True), use_container_width=True)
        st.markdown("**Sample of bandit experience**")
        st.dataframe(history_df.tail(15), use_container_width=True)

st.markdown("---")

# Video and selfie analysis
st.subheader("Session video and selfie analysis - prototype")
tab_vid, tab_img = st.tabs(["Video analysis", "Selfie analysis"])
with tab_vid:
    st.markdown("#### Upload work session video")
    video_file = st.file_uploader(
        "Video file",
        type=["mp4", "mov", "avi"],
        accept_multiple_files=False,
        key="video_upload",
    )
    if video_file is not None:
        st.video(video_file)
        st.warning(
            "This prototype does not yet run a deep vision model. "
            "In production, connect this module to a pose or focus detection model to score engagement."
        )
with tab_img:
    st.markdown("#### Upload selfie during session")
    img_file = st.file_uploader(
        "Image file",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="img_upload",
    )
    if img_file is not None:
        st.image(img_file, caption="Uploaded selfie", use_column_width=True)
        st.warning(
            "This prototype only previews the image. "
            "In production, plug in a face analysis model for emotion and fatigue signals following your HR policy."
        )
