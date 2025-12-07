import pathlib
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Basic config
# -----------------------------
st.set_page_config(
    page_title="HR AAC - AI Performance Lab",
    page_icon="📊",
    layout="wide",
)

st.title("📊 HR AAC - AI Performance Lab")
st.caption("Employee performance analytics with AI and reinforcement learning prototype")
st.markdown("> Prototype only - not for real HR decisions yet.")

# -----------------------------
# Data loading helpers
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
    # Fix common formatting mistakes
    t = t.replace("AMM", "AM").replace("PMM", "PM")
    # Add space before AM/PM if missing
    if t.lower().endswith("am") and not t.lower().endswith(" am"):
        t = t[:-2] + " AM"
    if t.lower().endswith("pm") and not t.lower().endswith(" pm"):
        t = t[:-2] + " PM"
    try:
        return pd.to_datetime(t, format="%I:%M %p", errors="coerce").time()
    except Exception:
        return pd.NaT


def enrich_with_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Parse times, compute work/break hours and task flags."""
    data = df.copy()

    # Normalize column names
    cols = {c: c.strip() for c in data.columns}
    data = data.rename(columns=cols)

    # Parse time columns if they exist
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

    # Task based metrics (robust to missing columns)
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

    # Date parsing
    if "Date" in data.columns:
        data["Date_parsed"] = pd.to_datetime(data["Date"], errors="coerce")
    else:
        data["Date_parsed"] = pd.NaT

    return data


def risk_flag_row(row):
    """Simple qualitative risk label based on performance score."""
    score = row.get("performance_score", 0)
    if score >= 80:
        return "🌟 Top performer"
    if score >= 60:
        return "✅ Stable"
    if score >= 40:
        return "⚠ Needs support"
    return "🚨 At risk"


def aggregate_perf(df_daily: pd.DataFrame, w_hours: float, w_tasks: float, w_progress: float) -> pd.DataFrame:
    """Aggregate daily metrics into per-employee performance with tunable weights."""
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

    # Aggregate by employee and date
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

    # Per employee summary
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

    # Normalize features
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
    """Very small hand crafted policy that mimics an RL agent decision."""
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
    """
    Simulated reward in [0, 1] for taking a management action on an employee.
    Uses performance_score and simple state; adds some noise.
    """
    base = (employee_row.get("performance_score", 50.0) or 50.0) / 100.0
    state = simple_state_from_row(employee_row)
    action = BANDIT_ACTIONS[action_index]

    bonus = 0.0

    # Simple heuristics
    if action == "Reward and promote" and state["tasks_level"] == "high":
        bonus += 0.15
    if action == "Provide targeted training" and state["tasks_level"] == "low":
        bonus += 0.15
    if action == "Reduce workload / focus work" and state["hours_level"] == "high" and state["tasks_level"] != "high":
        bonus += 0.1
    if action == "Increase check-ins and feedback" and state["break_level"] == "high":
        bonus += 0.1

    # Small randomness to mimic uncertainty
    noise = np.random.normal(0, 0.05)
    reward = base + bonus + noise
    return float(np.clip(reward, 0.0, 1.0))


def simulate_epsilon_greedy_bandit(
    df_perf: pd.DataFrame,
    episodes: int = 200,
    epsilon: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run an epsilon-greedy multi-armed bandit on top of employee data.
    Returns:
      history_df: per-episode history
      q_df: final action value estimates
    """
    if df_perf.empty:
        return pd.DataFrame(), pd.DataFrame()

    n_actions = len(BANDIT_ACTIONS)
    q_values = np.zeros(n_actions)
    counts = np.zeros(n_actions)

    history = []

    employees = df_perf["Team Members"].dropna().tolist()

    for ep in range(1, episodes + 1):
        # Sample a random employee as environment context
        emp_name = np.random.choice(employees)
        emp_row = df_perf[df_perf["Team Members"] == emp_name].iloc[0]

        # Epsilon greedy action choice
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(n_actions)
            greedy = False
        else:
            action_idx = int(np.argmax(q_values))
            greedy = True

        reward = bandit_reward_for(emp_row, action_idx)

        # Incremental update for Q-value of chosen action
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
# Sidebar - controls
# -----------------------------
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

# Enrich with metrics (work hours, flags, parsed dates)
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

# Build performance summary using filtered data
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

# -----------------------------
# Overview KPIs
# -----------------------------
st.subheader("Overview KPIs")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Tracked employees",
        f"{df_perf['Team Members'].nunique()}",
    )
with col2:
    avg_hours = df_perf["avg_daily_hours"].mean()
    st.metric(
        "Average daily hours",
        f"{avg_hours:0.2f} h" if not np.isnan(avg_hours) else "N/A",
    )
with col3:
    total_tasks = df_perf["total_tasks_completed"].sum()
    st.metric(
        "Tasks completed (total)",
        f"{int(total_tasks)}",
    )
with col4:
    high_perf = (df_perf["performance_score"] >= 70).sum()
    st.metric(
        "High performance profiles",
        f"{high_perf}",
    )

st.markdown("---")

# -----------------------------
# Single employee performance search
# -----------------------------
st.subheader("Single employee performance search")

search_query = st.text_input(
    "Type employee name (partial is OK)",
    placeholder="e.g., Arafat, Nayeem, Tanvir...",
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

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg daily hours", f"{emp_row['avg_daily_hours']:.2f} h")
    with c2:
        st.metric("Tasks completed", f"{int(emp_row['total_tasks_completed'])}")
    with c3:
        st.metric("Performance score", f"{emp_row['performance_score']:.1f} / 100")
    with c4:
        st.metric("Status", emp_row["risk_flag"])

    st.markdown("**Daily work hours timeline**")
    if not emp_daily.empty:
        emp_daily_plot = emp_daily[["Date_parsed", "work_hours"]].copy()
        emp_daily_plot = emp_daily_plot.set_index("Date_parsed")
        st.line_chart(emp_daily_plot["work_hours"])
    else:
        st.info("No daily records found for this employee.")

    # Export per-employee report
    emp_report = emp_daily.copy()
    emp_report["performance_score"] = emp_row["performance_score"]
    csv_data = emp_report.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {selected_emp}'s report (CSV)",
        data=csv_data,
        file_name=f"{selected_emp.replace(' ', '_')}_performance_report.csv",
        mime="text/csv",
    )

st.markdown("---")

# -----------------------------
# Charts for all employees / filtered
# -----------------------------
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

# -----------------------------
# Employee explorer
# -----------------------------
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

    member_for_ai = st.selectbox(
        "Choose an employee",
        team_members_all,
        key="ai_member",
    )

    row = df_perf[df_perf["Team Members"] == member_for_ai].iloc[0]
    st.markdown(
        f"""
        **{member_for_ai}**

        - Average daily hours: `{row['avg_daily_hours']:.2f}` h  
        - Average break hours: `{row['avg_break_hours']:.2f}` h  
        - Days observed: `{int(row['days_observed'])}` days  
        - Tasks completed: `{int(row['total_tasks_completed'])}`  
        - Tasks in progress: `{int(row['total_tasks_in_progress'])}`  
        - Composite performance score: `{row['performance_score']:.1f}` / 100  
        - Status: `{row['risk_flag']}`
        """
    )

    state = simple_state_from_row(row)
    action = rl_policy(state)

    st.markdown("**RL-style decision snapshot**")
    st.json(
        {
            "state": state,
            "chosen_action": action,
            "reward_intuition": (
                "Higher hours and completed tasks are rewarded. "
                "Excessive breaks with low task completion are penalized."
            ),
        }
    )

    st.success(f"Recommended management action: {action}")

st.markdown("---")

# -----------------------------
# RL sandbox - epsilon-greedy bandit
# -----------------------------
st.subheader("Reinforcement learning sandbox (epsilon-greedy bandit)")

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

# -----------------------------
# Video and selfie analysis - prototype
# -----------------------------
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

st.markdown(
    """
    To deploy on Streamlit Community Cloud or Streamlit Lite, 
    put `app.py`, the CSV, and a `requirements.txt` (`streamlit`, `pandas`, `numpy`) in a GitHub repo, 
    then point the deployment to `app.py`.
    """
)
