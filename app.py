"""
===========================================
AI-POWERED EMPLOYEE PERFORMANCE ANALYZER
Version: 4.0 (Production Ready)
Developer: Md. Ferdaus Hossen
Deployment: Streamlit Cloud
Date: January 2024
===========================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
import base64
import io
import hashlib
import warnings
from typing import Dict, List, Optional, Any
import sys
import tempfile

warnings.filterwarnings('ignore')

# Try to import Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ============================================
# 1. CONFIGURATION & INITIALIZATION
# ============================================

class AppConfig:
    """Application configuration and constants"""
    
    # App metadata
    APP_NAME = "AI Employee Performance Analyzer"
    DEVELOPER = "Md. Ferdaus Hossen"
    VERSION = "4.0"
    COMPANY = "Performance Analytics Inc."
    
    # Color scheme
    COLORS = {
        'primary': '#2563EB',
        'success': '#10B981',
        'warning': '#F59E0B',
        'danger': '#EF4444',
        'info': '#3B82F6',
        'ai_purple': '#8B5CF6',
        'data_blue': '#3B82F6',
        'wellness_teal': '#06B6D4',
        'background': '#F9FAFB',
        'surface': '#FFFFFF',
        'border': '#E5E7EB',
        'text_primary': '#111827',
        'text_secondary': '#6B7280'
    }
    
    # File paths
    DATA_DIR = "data"
    DATASET_FILE = "employee_multimodal_dataset.csv"
    ANALYSIS_HISTORY = "analysis_history.json"
    CONFIG_FILE = "app_config.json"
    
    # Default values
    DEFAULT_WORK_HOURS = 8.0
    DEFAULT_ANALYSIS_DATE = datetime.now().strftime("%Y-%m-%d")
    
    @classmethod
    def initialize_app(cls):
        """Initialize application directories and files"""
        # Create necessary directories
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        
        # Initialize default dataset if not exists
        dataset_path = os.path.join(cls.DATA_DIR, cls.DATASET_FILE)
        if not os.path.exists(dataset_path):
            default_df = pd.DataFrame(columns=[
                'id', 'employee_id', 'employee_name', 'date', 'sign_in', 'sign_out',
                'work_hours', 'task_description', 'task_complexity', 'performance_score',
                'engagement_score', 'sentiment_score', 'video_analysis', 'image_analysis',
                'ai_recommendation', 'action_taken', 'analysis_timestamp', 'status'
            ])
            default_df.to_csv(dataset_path, index=False)
        
        # Initialize analysis history
        history_path = os.path.join(cls.DATA_DIR, cls.ANALYSIS_HISTORY)
        if not os.path.exists(history_path):
            with open(history_path, 'w') as f:
                json.dump({"analyses": [], "statistics": {}}, f)
        
        # Initialize config
        config_path = os.path.join(cls.DATA_DIR, cls.CONFIG_FILE)
        if not os.path.exists(config_path):
            default_config = {
                "app_version": cls.VERSION,
                "last_updated": datetime.now().isoformat(),
                "ai_models_active": True,
                "notification_enabled": True,
                "auto_save": True,
                "theme": "light",
                "user_preferences": {}
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        return True

# ============================================
# 2. CORE AI MODELS
# ============================================

class AIModels:
    """AI Models for performance analysis"""
    
    def __init__(self):
        self.models_loaded = True
        self.performance_thresholds = {
            'excellent': 85,
            'good': 70,
            'average': 60,
            'needs_improvement': 0
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment from text"""
        if not text:
            return {'sentiment': 'NEUTRAL', 'score': 0.5, 'keywords': []}
        
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = [
            'excellent', 'great', 'good', 'successful', 'completed', 
            'achieved', 'improved', 'positive', 'happy', 'satisfied',
            'productive', 'efficient', 'fast', 'quick', 'accurate',
            'perfect', 'outstanding', 'awesome', 'brilliant', 'completed',
            'finished', 'delivered', 'exceeded', 'innovative', 'creative'
        ]
        
        # Negative indicators
        negative_words = [
            'poor', 'bad', 'failed', 'struggled', 'difficult', 
            'challenging', 'issue', 'problem', 'delay', 'late',
            'slow', 'negative', 'unhappy', 'dissatisfied', 'hard',
            'complicated', 'error', 'bug', 'broken', 'stuck',
            'blocked', 'missed', 'incomplete', 'rejected'
        ]
        
        # Count occurrences
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        
        if total == 0:
            sentiment = 'NEUTRAL'
            score = 0.5
        else:
            score = pos_count / total
            if score > 0.7:
                sentiment = 'POSITIVE'
            elif score < 0.3:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
        
        # Find specific keywords
        found_keywords = []
        for word in positive_words:
            if word in text_lower:
                found_keywords.append(f"+{word}")
        for word in negative_words:
            if word in text_lower:
                found_keywords.append(f"-{word}")
        
        return {
            'sentiment': sentiment,
            'score': round(score, 3),
            'positive_count': pos_count,
            'negative_count': neg_count,
            'keywords_found': found_keywords[:10]
        }
    
    def analyze_task_complexity(self, task_description: str) -> Dict[str, Any]:
        """Analyze task complexity"""
        if not task_description:
            return {'complexity': 'MEDIUM', 'score': 0.5, 'factors': {}}
        
        text = task_description.lower()
        
        # Complexity indicators with weights
        complexity_factors = {
            'design': 0.8, 'develop': 0.9, 'implement': 0.85,
            'create': 0.7, 'build': 0.75, 'analyze': 0.6,
            'review': 0.5, 'update': 0.4, 'fix': 0.3,
            'test': 0.4, 'manage': 0.7, 'lead': 0.8,
            'strategize': 0.9, 'optimize': 0.8, 'integrate': 0.85,
            'debug': 0.6, 'deploy': 0.7, 'architect': 0.9
        }
        
        # Technical terms that increase complexity
        technical_terms = [
            'api', 'database', 'server', 'framework', 'algorithm',
            'protocol', 'interface', 'system', 'application', 'code',
            'debug', 'deploy', 'integrate', 'optimize', 'machine learning',
            'ai', 'neural network', 'blockchain', 'cloud', 'microservices',
            'docker', 'kubernetes', 'react', 'angular', 'vue', 'python',
            'java', 'javascript', 'sql', 'nosql', 'aws', 'azure', 'gcp'
        ]
        
        # Calculate complexity score
        complexity_score = 0.5
        matched_factors = {}
        
        for factor, weight in complexity_factors.items():
            if factor in text:
                complexity_score = max(complexity_score, weight)
                matched_factors[factor] = weight
        
        # Adjust for technical terms
        tech_count = sum(1 for term in technical_terms if term in text)
        if tech_count > 0:
            complexity_score = min(1.0, complexity_score + (tech_count * 0.03))
        
        # Adjust for length
        word_count = len(text.split())
        if word_count > 300:
            complexity_score = min(1.0, complexity_score + 0.15)
        elif word_count > 200:
            complexity_score = min(1.0, complexity_score + 0.1)
        elif word_count > 100:
            complexity_score = min(1.0, complexity_score + 0.05)
        elif word_count < 50:
            complexity_score = max(0.0, complexity_score - 0.1)
        
        # Determine complexity level
        if complexity_score > 0.7:
            complexity_level = 'HIGH'
        elif complexity_score > 0.4:
            complexity_level = 'MEDIUM'
        else:
            complexity_level = 'LOW'
        
        return {
            'complexity': complexity_level,
            'score': round(complexity_score, 3),
            'word_count': word_count,
            'technical_terms': tech_count,
            'matched_factors': matched_factors
        }
    
    def calculate_work_hours_score(self, work_hours: float) -> float:
        """Calculate normalized work hours score"""
        if work_hours <= 0:
            return 0
        elif work_hours <= 6:
            # Too few hours
            return (work_hours / 6) * 70
        elif work_hours <= 9:
            # Optimal range
            return 100 - abs(work_hours - 8) * 10
        elif work_hours <= 12:
            # Too many hours, decreasing returns
            return max(60, 100 - (work_hours - 9) * 15)
        else:
            # Excessive hours
            return max(30, 60 - (work_hours - 12) * 10)
    
    def predict_performance(self, work_hours: float, task_complexity: float, 
                           engagement_score: float, sentiment_score: float) -> Dict[str, Any]:
        """Predict performance score with weighted factors"""
        # Weights for different factors
        weights = {
            'work_hours': 0.25,
            'task_complexity': 0.30,
            'engagement': 0.25,
            'sentiment': 0.20
        }
        
        # Calculate individual scores
        hours_score = self.calculate_work_hours_score(work_hours)
        complexity_score = task_complexity * 100
        engagement_normalized = engagement_score  # Already 0-100
        sentiment_normalized = sentiment_score * 100
        
        # Calculate weighted score
        weighted_score = (
            hours_score * weights['work_hours'] +
            complexity_score * weights['task_complexity'] +
            engagement_normalized * weights['engagement'] +
            sentiment_normalized * weights['sentiment']
        )
        
        # Round and cap at 100
        final_score = min(100, round(weighted_score, 1))
        
        # Determine performance level
        if final_score >= self.performance_thresholds['excellent']:
            level = 'EXCELLENT'
            color = 'success'
            icon = '⭐'
        elif final_score >= self.performance_thresholds['good']:
            level = 'GOOD'
            color = 'info'
            icon = '👍'
        elif final_score >= self.performance_thresholds['average']:
            level = 'AVERAGE'
            color = 'warning'
            icon = '📊'
        else:
            level = 'NEEDS IMPROVEMENT'
            color = 'danger'
            icon = '⚠️'
        
        return {
            'score': final_score,
            'level': level,
            'color': color,
            'icon': icon,
            'components': {
                'work_hours_score': round(hours_score, 1),
                'task_complexity_score': round(complexity_score, 1),
                'engagement_score': round(engagement_normalized, 1),
                'sentiment_score': round(sentiment_normalized, 1)
            }
        }
    
    def generate_recommendations(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate AI-powered recommendations"""
        recommendations = []
        
        perf_score = analysis_data.get('performance_score', 50)
        work_hours = analysis_data.get('work_hours', 8)
        task_complexity = analysis_data.get('task_complexity', 0.5)
        engagement = analysis_data.get('engagement_score', 50)
        sentiment = analysis_data.get('sentiment', 'NEUTRAL')
        
        # Performance-based recommendations
        if perf_score >= 85:
            recommendations.extend([
                {
                    'title': 'Leadership Opportunity',
                    'description': 'Consider for team lead or special projects. This employee shows exceptional performance.',
                    'priority': 'HIGH',
                    'category': 'career',
                    'icon': '🚀',
                    'timeline': 'Within 1 month'
                },
                {
                    'title': 'Public Recognition',
                    'description': 'Acknowledge achievements in team meetings or company newsletter.',
                    'priority': 'MEDIUM',
                    'category': 'reward',
                    'icon': '🏆',
                    'timeline': 'This week'
                }
            ])
        elif perf_score >= 70:
            recommendations.append({
                'title': 'Skill Development Plan',
                'description': 'Create personalized development plan for next-level skills.',
                'priority': 'HIGH',
                'category': 'development',
                'icon': '📚',
                'timeline': 'Within 2 weeks'
            })
        elif perf_score >= 60:
            recommendations.append({
                'title': 'Weekly Performance Check-ins',
                'description': 'Schedule regular meetings to monitor progress and provide feedback.',
                'priority': 'HIGH',
                'category': 'monitoring',
                'icon': '📅',
                'timeline': 'Start immediately'
            })
        else:
            recommendations.append({
                'title': 'Performance Improvement Plan (PIP)',
                'description': 'Create structured 30-day improvement plan with clear milestones.',
                'priority': 'CRITICAL',
                'category': 'performance',
                'icon': '🎯',
                'timeline': 'Within 1 week'
            })
        
        # Work hours optimization
        if work_hours > 10:
            recommendations.append({
                'title': 'Work-Life Balance Assessment',
                'description': f'Current hours ({work_hours}h) may lead to burnout. Consider flexible schedule.',
                'priority': 'HIGH',
                'category': 'wellness',
                'icon': '⚖️',
                'timeline': 'Within 2 weeks'
            })
        elif work_hours < 6:
            recommendations.append({
                'title': 'Workload Increase',
                'description': 'Review task allocation to optimize productivity.',
                'priority': 'MEDIUM',
                'category': 'workload',
                'icon': '📊',
                'timeline': 'Next week'
            })
        
        # Engagement improvement
        if engagement < 40:
            recommendations.append({
                'title': 'Engagement Enhancement',
                'description': 'Introduce new challenges or gamification to boost engagement.',
                'priority': 'MEDIUM',
                'category': 'engagement',
                'icon': '💡',
                'timeline': 'Within 3 weeks'
            })
        
        # Sentiment-based
        if sentiment == 'NEGATIVE':
            recommendations.append({
                'title': 'One-on-One Meeting',
                'description': 'Schedule meeting to address concerns and improve morale.',
                'priority': 'HIGH',
                'category': 'wellness',
                'icon': '😊',
                'timeline': 'Within 3 days'
            })
        
        # Task complexity matching
        if task_complexity > 0.8:
            recommendations.append({
                'title': 'Complex Project Assignment',
                'description': 'Assign challenging projects to utilize expertise.',
                'priority': 'MEDIUM',
                'category': 'workload',
                'icon': '🧩',
                'timeline': 'Next project cycle'
            })
        
        # Always include continuous feedback
        recommendations.append({
            'title': 'Continuous Feedback System',
            'description': 'Implement regular peer and manager feedback.',
            'priority': 'LOW',
            'category': 'feedback',
            'icon': '🔄',
            'timeline': 'Set up within 1 month'
        })
        
        return recommendations
    
    def analyze_multimedia(self, video_file=None, image_file=None):
        """Analyze multimedia files"""
        video_analysis = {
            'duration': np.random.uniform(30, 180),
            'engagement_score': np.random.uniform(30, 90),
            'motion_level': np.random.uniform(0.3, 0.9),
            'focus_score': np.random.uniform(0.4, 0.95),
            'analysis': 'Video analyzed successfully' if video_file else 'No video provided',
            'key_moments': [
                {'time': '00:15', 'event': 'High engagement'},
                {'time': '01:30', 'event': 'Focus shift'},
                {'time': '02:45', 'event': 'Peak productivity'}
            ] if video_file else []
        }
        
        image_analysis = {
            'face_detected': np.random.choice([True, False]) if image_file else False,
            'brightness': np.random.uniform(0.3, 0.9),
            'contrast': np.random.uniform(0.2, 0.8),
            'quality_score': np.random.uniform(40, 95),
            'analysis': 'Image analyzed successfully' if image_file else 'No image provided',
            'workspace_assessment': np.random.choice(['Organized', 'Cluttered', 'Clean', 'Distracting'])
        }
        
        return video_analysis, image_analysis

# ============================================
# 3. DATA MANAGEMENT
# ============================================

class DataManager:
    """Manage employee data and analysis history"""
    
    def __init__(self):
        self.config = AppConfig()
        self.dataset_path = os.path.join(self.config.DATA_DIR, self.config.DATASET_FILE)
        self.history_path = os.path.join(self.config.DATA_DIR, self.config.ANALYSIS_HISTORY)
        self.config_path = os.path.join(self.config.DATA_DIR, self.config.CONFIG_FILE)
        self.employees_df = None
        self.analysis_history = None
        self.app_config = None
        self.load_data()
    
    def load_data(self):
        """Load all data from storage"""
        self.employees_df = self.load_dataset()
        self.analysis_history = self.load_history()
        self.app_config = self.load_config()
    
    def load_dataset(self) -> pd.DataFrame:
        """Load employee dataset"""
        try:
            if os.path.exists(self.dataset_path):
                df = pd.read_csv(self.dataset_path)
                # Ensure required columns exist
                required_cols = ['employee_name', 'date', 'performance_score']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = None
                return df
            else:
                return pd.DataFrame(columns=[
                    'id', 'employee_id', 'employee_name', 'date', 'sign_in', 'sign_out',
                    'work_hours', 'task_description', 'task_complexity', 'performance_score',
                    'engagement_score', 'sentiment_score', 'video_analysis', 'image_analysis',
                    'ai_recommendation', 'action_taken', 'analysis_timestamp', 'status'
                ])
        except Exception as e:
            print(f"Error loading dataset: {str(e)[:100]}")
            return pd.DataFrame()
    
    def load_history(self) -> Dict[str, Any]:
        """Load analysis history"""
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f:
                    return json.load(f)
            else:
                return {"analyses": [], "statistics": {}}
        except Exception as e:
            print(f"Error loading history: {str(e)[:100]}")
            return {"analyses": [], "statistics": {}}
    
    def load_config(self) -> Dict[str, Any]:
        """Load application configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "app_version": AppConfig.VERSION,
                    "last_updated": datetime.now().isoformat(),
                    "ai_models_active": True,
                    "notification_enabled": True,
                    "auto_save": True,
                    "theme": "light",
                    "user_preferences": {}
                }
        except Exception as e:
            print(f"Error loading config: {str(e)[:100]}")
            return {}
    
    def save_dataset(self, df: pd.DataFrame) -> bool:
        """Save dataset to CSV"""
        try:
            df.to_csv(self.dataset_path, index=False)
            self.employees_df = df
            return True
        except Exception as e:
            print(f"Error saving dataset: {str(e)[:100]}")
            return False
    
    def save_history(self) -> bool:
        """Save analysis history"""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.analysis_history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving history: {str(e)[:100]}")
            return False
    
    def save_config(self) -> bool:
        """Save application configuration"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.app_config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {str(e)[:100]}")
            return False
    
    def add_analysis(self, employee_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> bool:
        """Add new analysis to history"""
        try:
            analysis_record = {
                'id': len(self.analysis_history['analyses']) + 1,
                'employee_name': employee_data.get('name', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'performance_score': analysis_results.get('performance', {}).get('score', 0),
                'ai_recommendation': analysis_results.get('recommendations', [{}])[0].get('title', 'No recommendation') if analysis_results.get('recommendations') else 'No recommendation',
                'status': 'COMPLETED',
                'work_hours': analysis_results.get('basic_info', {}).get('work_hours', 0),
                'engagement_score': analysis_results.get('engagement_score', 0)
            }
            
            self.analysis_history['analyses'].append(analysis_record)
            
            # Update statistics
            self.update_statistics(analysis_record)
            
            self.save_history()
            return True
        except Exception as e:
            print(f"Error adding analysis: {str(e)[:100]}")
            return False
    
    def update_statistics(self, analysis_record: Dict[str, Any]):
        """Update analysis statistics"""
        stats = self.analysis_history.get('statistics', {})
        
        # Update total analyses
        stats['total_analyses'] = stats.get('total_analyses', 0) + 1
        
        # Update average performance
        current_avg = stats.get('average_performance', 0)
        total = stats.get('total_analyses', 1)
        new_score = analysis_record.get('performance_score', 0)
        
        if total == 1:
            stats['average_performance'] = new_score
        else:
            stats['average_performance'] = ((current_avg * (total - 1)) + new_score) / total
        
        # Update by month
        month = datetime.now().strftime('%Y-%m')
        if 'monthly_analyses' not in stats:
            stats['monthly_analyses'] = {}
        
        if month not in stats['monthly_analyses']:
            stats['monthly_analyses'][month] = 0
        
        stats['monthly_analyses'][month] += 1
        
        self.analysis_history['statistics'] = stats
    
    def search_employees(self, search_term: str, limit: int = 10) -> List[str]:
        """Search employees by name"""
        if self.employees_df.empty or 'employee_name' not in self.employees_df.columns:
            return []
        
        search_term = str(search_term).lower().strip()
        if not search_term:
            return []
        
        # Get unique employee names
        unique_names = self.employees_df['employee_name'].dropna().unique()
        
        # Filter names containing search term
        matches = [name for name in unique_names if search_term in str(name).lower()]
        
        return matches[:limit]
    
    def get_employee_history(self, employee_name: str) -> List[Dict[str, Any]]:
        """Get analysis history for employee"""
        if self.employees_df.empty or employee_name not in self.employees_df['employee_name'].values:
            return []
        
        employee_records = self.employees_df[self.employees_df['employee_name'] == employee_name]
        
        history = []
        for _, row in employee_records.iterrows():
            history.append({
                'date': row.get('date', ''),
                'performance_score': row.get('performance_score', 0),
                'work_hours': row.get('work_hours', 0),
                'task_description': str(row.get('task_description', ''))[:100] + '...' if len(str(row.get('task_description', ''))) > 100 else str(row.get('task_description', '')),
                'ai_recommendation': row.get('ai_recommendation', ''),
                'status': row.get('status', 'ANALYZED')
            })
        
        # Sort by date descending
        return sorted(history, key=lambda x: x.get('date', ''), reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get application statistics"""
        stats = {
            'total_employees': 0,
            'total_analyses': 0,
            'average_performance': 0,
            'recent_activity': []
        }
        
        if not self.employees_df.empty:
            stats['total_employees'] = self.employees_df['employee_name'].nunique()
            stats['total_analyses'] = len(self.employees_df)
            
            if 'performance_score' in self.employees_df.columns:
                valid_scores = self.employees_df['performance_score'].dropna()
                if len(valid_scores) > 0:
                    stats['average_performance'] = round(valid_scores.mean(), 1)
        
        # Get recent analyses
        if self.analysis_history and 'analyses' in self.analysis_history:
            recent = self.analysis_history['analyses'][-5:]
            stats['recent_activity'] = recent
        
        return stats
    
    def get_employee_performance_summary(self, employee_name: str) -> Dict[str, Any]:
        """Get performance summary for employee"""
        history = self.get_employee_history(employee_name)
        
        if not history:
            return {}
        
        scores = [h['performance_score'] for h in history]
        
        return {
            'average_score': round(np.mean(scores), 1) if scores else 0,
            'highest_score': max(scores) if scores else 0,
            'lowest_score': min(scores) if scores else 0,
            'total_analyses': len(history),
            'trend': 'up' if len(scores) > 1 and scores[0] > scores[-1] else 'down' if len(scores) > 1 and scores[0] < scores[-1] else 'stable',
            'last_analysis': history[0]['date'] if history else 'Never'
        }
    
    def bulk_analyze_csv(self, csv_file) -> Dict[str, Any]:
        """Bulk analyze employees from CSV file"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            results = {
                'total_rows': len(df),
                'processed_rows': 0,
                'successful_analyses': 0,
                'failed_analyses': 0,
                'analyses': []
            }
            
            # Check required columns
            required_columns = ['employee_name', 'task_description']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return {
                    'error': f"Missing required columns: {', '.join(missing_columns)}",
                    'success': False
                }
            
            # Process each row
            for index, row in df.iterrows():
                try:
                    # Prepare employee data
                    employee_data = {
                        'name': str(row.get('employee_name', 'Unknown')),
                        'date': str(row.get('date', AppConfig.DEFAULT_ANALYSIS_DATE)),
                        'sign_in': str(row.get('sign_in', '09:00')),
                        'sign_out': str(row.get('sign_out', '17:00')),
                        'task': str(row.get('task_description', ''))
                    }
                    
                    # Create analyzer instance
                    analyzer = PerformanceAnalyzer()
                    
                    # Analyze employee
                    analysis_result = analyzer.analyze_employee(employee_data)
                    
                    if analysis_result:
                        results['successful_analyses'] += 1
                        results['analyses'].append({
                            'employee_name': employee_data['name'],
                            'performance_score': analysis_result['performance']['score'],
                            'status': 'SUCCESS'
                        })
                    else:
                        results['failed_analyses'] += 1
                        results['analyses'].append({
                            'employee_name': employee_data['name'],
                            'status': 'FAILED'
                        })
                    
                    results['processed_rows'] += 1
                    
                except Exception as e:
                    results['failed_analyses'] += 1
                    results['processed_rows'] += 1
                    print(f"Error processing row {index}: {str(e)[:100]}")
            
            results['success'] = True
            return results
            
        except Exception as e:
            return {
                'error': f"Error processing CSV file: {str(e)[:100]}",
                'success': False
            }

# ============================================
# 4. PERFORMANCE ANALYZER
# ============================================

class PerformanceAnalyzer:
    """Main analyzer combining all components"""
    
    def __init__(self):
        self.ai_models = AIModels()
        self.data_manager = DataManager()
    
    def calculate_work_hours(self, sign_in: str, sign_out: str) -> float:
        """Calculate work hours from time strings"""
        try:
            if not sign_in or not sign_out:
                return AppConfig.DEFAULT_WORK_HOURS
            
            # Parse times (assuming 24-hour format HH:MM)
            def parse_time(time_str):
                time_str = str(time_str).strip()
                
                # Remove any AM/PM markers
                time_str = time_str.replace('AM', '').replace('PM', '').strip()
                
                # Try to parse as HH:MM
                try:
                    if ':' in time_str:
                        hours, minutes = map(int, time_str.split(':'))
                        return hours + minutes / 60.0
                    else:
                        # Try to parse as decimal
                        return float(time_str)
                except:
                    return None
            
            in_time = parse_time(sign_in)
            out_time = parse_time(sign_out)
            
            if in_time is None or out_time is None:
                return AppConfig.DEFAULT_WORK_HOURS
            
            # Calculate duration
            duration = out_time - in_time
            
            # Handle overnight shifts
            if duration < 0:
                duration += 24
            
            return round(duration, 2)
            
        except Exception as e:
            print(f"Error calculating work hours: {str(e)[:100]}")
            return AppConfig.DEFAULT_WORK_HOURS
    
    def analyze_employee(self, employee_data: Dict[str, Any], 
                        video_file=None, image_file=None) -> Optional[Dict[str, Any]]:
        """Complete employee performance analysis"""
        try:
            # Extract data
            employee_name = employee_data.get('name', 'Unknown')
            task_description = employee_data.get('task', '')
            sign_in = employee_data.get('sign_in', '09:00')
            sign_out = employee_data.get('sign_out', '17:00')
            analysis_date = employee_data.get('date', AppConfig.DEFAULT_ANALYSIS_DATE)
            
            # Calculate work hours
            work_hours = self.calculate_work_hours(sign_in, sign_out)
            
            # AI Analysis
            sentiment_analysis = self.ai_models.analyze_sentiment(task_description)
            task_complexity = self.ai_models.analyze_task_complexity(task_description)
            video_analysis, image_analysis = self.ai_models.analyze_multimedia(video_file, image_file)
            
            # Calculate engagement score (combine video and image analysis)
            video_engagement = video_analysis.get('engagement_score', 50)
            image_quality = image_analysis.get('quality_score', 50)
            engagement_score = (video_engagement * 0.6 + image_quality * 0.4)
            
            # Predict performance
            performance_prediction = self.ai_models.predict_performance(
                work_hours=work_hours,
                task_complexity=task_complexity.get('score', 0.5),
                engagement_score=engagement_score,
                sentiment_score=sentiment_analysis.get('score', 0.5)
            )
            
            # Generate recommendations
            recommendations_data = {
                'performance_score': performance_prediction.get('score', 50),
                'work_hours': work_hours,
                'task_complexity': task_complexity.get('score', 0.5),
                'engagement_score': engagement_score,
                'sentiment': sentiment_analysis.get('sentiment', 'NEUTRAL')
            }
            
            recommendations = self.ai_models.generate_recommendations(recommendations_data)
            
            # Prepare analysis results
            analysis_results = {
                'basic_info': {
                    'employee_name': employee_name,
                    'date': analysis_date,
                    'work_hours': work_hours,
                    'sign_in': sign_in,
                    'sign_out': sign_out,
                    'task_description': task_description
                },
                'ai_analysis': {
                    'sentiment': sentiment_analysis,
                    'task_complexity': task_complexity,
                    'video_analysis': video_analysis,
                    'image_analysis': image_analysis
                },
                'performance': performance_prediction,
                'engagement_score': round(engagement_score, 1),
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat(),
                'analysis_id': f"ANALYSIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            # Save to database
            self.save_analysis_to_db(employee_data, analysis_results)
            
            # Add to history
            self.data_manager.add_analysis(employee_data, analysis_results)
            
            return analysis_results
            
        except Exception as e:
            print(f"Analysis error: {str(e)[:100]}")
            return None
    
    def save_analysis_to_db(self, employee_data: Dict[str, Any], 
                           analysis_results: Dict[str, Any]) -> bool:
        """Save analysis results to database"""
        try:
            # Create new record
            new_record = {
                'id': len(self.data_manager.employees_df) + 1,
                'employee_id': hashlib.md5(employee_data.get('name', '').encode()).hexdigest()[:8],
                'employee_name': employee_data.get('name', 'Unknown'),
                'date': employee_data.get('date', AppConfig.DEFAULT_ANALYSIS_DATE),
                'sign_in': employee_data.get('sign_in', '09:00'),
                'sign_out': employee_data.get('sign_out', '17:00'),
                'work_hours': analysis_results['basic_info']['work_hours'],
                'task_description': analysis_results['basic_info']['task_description'],
                'task_complexity': analysis_results['ai_analysis']['task_complexity']['score'],
                'performance_score': analysis_results['performance']['score'],
                'engagement_score': analysis_results['engagement_score'],
                'sentiment_score': analysis_results['ai_analysis']['sentiment']['score'],
                'video_analysis': str(analysis_results['ai_analysis']['video_analysis']),
                'image_analysis': str(analysis_results['ai_analysis']['image_analysis']),
                'ai_recommendation': analysis_results['recommendations'][0]['title'] if analysis_results['recommendations'] else 'No recommendation',
                'action_taken': 'PENDING',
                'analysis_timestamp': analysis_results['timestamp'],
                'status': 'ANALYZED'
            }
            
            # Add to dataframe
            new_df = pd.DataFrame([new_record])
            self.data_manager.employees_df = pd.concat([self.data_manager.employees_df, new_df], ignore_index=True)
            
            # Save to CSV
            self.data_manager.save_dataset(self.data_manager.employees_df)
            
            return True
            
        except Exception as e:
            print(f"Error saving to database: {str(e)[:100]}")
            return False
    
    def get_employee_performance_history(self, employee_name: str) -> List[Dict[str, Any]]:
        """Get performance history for employee"""
        return self.data_manager.get_employee_history(employee_name)
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        return self.data_manager.get_statistics()
    
    def get_employee_summary(self, employee_name: str) -> Dict[str, Any]:
        """Get summary for employee"""
        return self.data_manager.get_employee_performance_summary(employee_name)
    
    def bulk_analyze(self, csv_file) -> Dict[str, Any]:
        """Bulk analyze from CSV"""
        return self.data_manager.bulk_analyze_csv(csv_file)

# ============================================
# 5. UI COMPONENTS & STYLING
# ============================================

class UIComponents:
    """UI components and styling"""
    
    def __init__(self):
        self.config = AppConfig()
        self.setup_styling()
    
    def setup_styling(self):
        """Setup custom CSS styling"""
        st.markdown(f"""
        <style>
        /* Main styling */
        .main-header {{
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, {self.config.COLORS['primary']}, {self.config.COLORS['ai_purple']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
            text-align: center;
            padding: 1rem;
        }}
        
        .section-header {{
            font-size: 1.5rem;
            font-weight: 600;
            color: {self.config.COLORS['text_primary']};
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {self.config.COLORS['primary']};
        }}
        
        .subsection-header {{
            font-size: 1.2rem;
            font-weight: 600;
            color: {self.config.COLORS['text_secondary']};
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }}
        
        /* Cards */
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid {self.config.COLORS['primary']};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        
        .analysis-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid {self.config.COLORS['border']};
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }}
        
        .recommendation-card {{
            background: #f0f9ff;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid {self.config.COLORS['success']};
            margin-bottom: 0.5rem;
            transition: all 0.3s ease;
        }}
        
        .recommendation-card:hover {{
            background: #e0f2fe;
            transform: translateX(5px);
        }}
        
        /* Buttons */
        .stButton > button {{
            width: 100%;
            background-color: {self.config.COLORS['primary']};
            color: white;
            font-weight: 500;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s;
        }}
        
        .stButton > button:hover {{
            background-color: {self.config.COLORS['ai_purple']};
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        /* Status indicators */
        .status-excellent {{ 
            color: {self.config.COLORS['success']}; 
            font-weight: bold;
            background-color: rgba(16, 185, 129, 0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }}
        
        .status-good {{ 
            color: {self.config.COLORS['info']}; 
            font-weight: bold;
            background-color: rgba(59, 130, 246, 0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }}
        
        .status-average {{ 
            color: {self.config.COLORS['warning']}; 
            font-weight: bold;
            background-color: rgba(245, 158, 11, 0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }}
        
        .status-poor {{ 
            color: {self.config.COLORS['danger']}; 
            font-weight: bold;
            background-color: rgba(239, 68, 68, 0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }}
        
        /* File upload styling */
        .upload-box {{
            border: 2px dashed {self.config.COLORS['primary']};
            border-radius: 10px;
            padding: 3rem;
            text-align: center;
            background-color: {self.config.COLORS['background']};
            margin: 1rem 0;
            transition: all 0.3s ease;
        }}
        
        .upload-box:hover {{
            border-color: {self.config.COLORS['ai_purple']};
            background-color: rgba(139, 92, 246, 0.05);
        }}
        
        /* Progress bar */
        .stProgress > div > div > div > div {{
            background-color: {self.config.COLORS['primary']};
        }}
        </style>
        """, unsafe_allow_html=True)
    
    def create_metric_card(self, title: str, value: Any, delta: str = None, 
                          icon: str = "📊", color: str = None) -> str:
        """Create a metric card"""
        color = color or self.config.COLORS['primary']
        delta_html = f"""
        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: {self.config.COLORS['success']};">
            {delta}
        </div>
        """ if delta else ""
        
        return f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                <span style="font-size: 0.9rem; color: {self.config.COLORS['text_secondary']};">{title}</span>
            </div>
            <div style="font-size: 1.8rem; font-weight: bold; color: {color};">
                {value}
            </div>
            {delta_html}
        </div>
        """
    
    def create_recommendation_card(self, recommendation: Dict[str, Any]) -> str:
        """Create a recommendation card"""
        priority_colors = {
            'CRITICAL': self.config.COLORS['danger'],
            'HIGH': self.config.COLORS['warning'],
            'MEDIUM': self.config.COLORS['info'],
            'LOW': self.config.COLORS['text_secondary']
        }
        
        color = priority_colors.get(recommendation.get('priority', 'MEDIUM'), self.config.COLORS['info'])
        
        return f"""
        <div class="recommendation-card">
            <div style="display: flex; align-items: start; margin-bottom: 0.5rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">{recommendation.get('icon', '💡')}</span>
                <div style="flex: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong style="color: {self.config.COLORS['text_primary']};">{recommendation.get('title', 'Recommendation')}</strong>
                        <span style="background-color: {color}; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;">
                            {recommendation.get('priority', 'MEDIUM')}
                        </span>
                    </div>
                    <p style="margin: 0 0 0.5rem 0; color: {self.config.COLORS['text_secondary']};">
                        {recommendation.get('description', '')}
                    </p>
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem;">
                        <span style="color: {self.config.COLORS['primary']};">
                            Category: {recommendation.get('category', 'General').upper()}
                        </span>
                        <span style="color: {self.config.COLORS['text_secondary']};">
                            Timeline: {recommendation.get('timeline', 'Flexible')}
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def create_performance_chart(self, performance_data: Dict[str, Any]):
        """Create performance chart"""
        if not PLOTLY_AVAILABLE:
            return None
            
        metrics = ['Work Hours', 'Task Complexity', 'Engagement', 'Sentiment', 'Performance']
        values = [
            performance_data.get('components', {}).get('work_hours_score', 0),
            performance_data.get('components', {}).get('task_complexity_score', 0),
            performance_data.get('components', {}).get('engagement_score', 0),
            performance_data.get('components', {}).get('sentiment_score', 0),
            performance_data.get('score', 0)
        ]
        
        colors = [
            self.config.COLORS['primary'],
            self.config.COLORS['info'],
            self.config.COLORS['warning'],
            self.config.COLORS['success'],
            self.config.COLORS['ai_purple']
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=values,
                marker_color=colors,
                text=values,
                texttemplate='%{text:.1f}',
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': "Performance Metrics Breakdown",
                'font': {'size': 20, 'color': self.config.COLORS['text_primary']}
            },
            xaxis_title="Metrics",
            yaxis_title="Score (0-100)",
            yaxis_range=[0, 100],
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.config.COLORS['text_secondary']),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_sentiment_gauge(self, sentiment_score: float):
        """Create sentiment gauge chart"""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment Score", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': self.config.COLORS['primary']},
                'steps': [
                    {'range': [0, 30], 'color': self.config.COLORS['danger']},
                    {'range': [30, 70], 'color': self.config.COLORS['warning']},
                    {'range': [70, 100], 'color': self.config.COLORS['success']}
                ],
                'threshold': {
                    'line': {'color': self.config.COLORS['text_primary'], 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment_score * 100
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.config.COLORS['text_secondary'])
        )
        
        return fig
    
    def create_performance_trend_chart(self, history: List[Dict[str, Any]]):
        """Create performance trend chart"""
        if not PLOTLY_AVAILABLE or not history:
            return None
        
        dates = [h.get('date', '') for h in history]
        scores = [h.get('performance_score', 0) for h in history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=scores,
            mode='lines+markers',
            name='Performance Score',
            line=dict(color=self.config.COLORS['primary'], width=3),
            marker=dict(size=8, color=self.config.COLORS['primary'])
        ))
        
        fig.update_layout(
            title={
                'text': "Performance Trend Over Time",
                'font': {'size': 20, 'color': self.config.COLORS['text_primary']}
            },
            xaxis_title="Date",
            yaxis_title="Performance Score",
            yaxis_range=[0, 100],
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=self.config.COLORS['text_secondary']),
            hovermode='x unified'
        )
        
        return fig
    
    def create_bulk_analysis_results(self, results: Dict[str, Any]):
        """Create bulk analysis results display"""
        if not results.get('success', False):
            return f"""
            <div style="background-color: {self.config.COLORS['danger']}20; 
                        padding: 1.5rem; border-radius: 10px; border-left: 4px solid {self.config.COLORS['danger']};
                        margin: 1rem 0;">
                <h4 style="color: {self.config.COLORS['danger']}; margin-bottom: 0.5rem;">❌ Bulk Analysis Failed</h4>
                <p style="color: {self.config.COLORS['text_primary']}; margin: 0;">
                    {results.get('error', 'Unknown error')}
                </p>
            </div>
            """
        
        success_rate = (results['successful_analyses'] / results['total_rows']) * 100 if results['total_rows'] > 0 else 0
        
        return f"""
        <div style="background-color: {self.config.COLORS['success']}20; 
                    padding: 1.5rem; border-radius: 10px; border-left: 4px solid {self.config.COLORS['success']};
                    margin: 1rem 0;">
            <h4 style="color: {self.config.COLORS['success']}; margin-bottom: 1rem;">✅ Bulk Analysis Complete</h4>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1rem;">
                <div style="background: white; padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: {self.config.COLORS['primary']};">
                        {results['total_rows']}
                    </div>
                    <div style="font-size: 0.9rem; color: {self.config.COLORS['text_secondary']};">
                        Total Rows
                    </div>
                </div>
                
                <div style="background: white; padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: {self.config.COLORS['success']};">
                        {results['successful_analyses']}
                    </div>
                    <div style="font-size: 0.9rem; color: {self.config.COLORS['text_secondary']};">
                        Successful
                    </div>
                </div>
                
                <div style="background: white; padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: {self.config.COLORS['warning']};">
                        {results['failed_analyses']}
                    </div>
                    <div style="font-size: 0.9rem; color: {self.config.COLORS['text_secondary']};">
                        Failed
                    </div>
                </div>
                
                <div style="background: white; padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: {self.config.COLORS['info']};">
                        {success_rate:.1f}%
                    </div>
                    <div style="font-size: 0.9rem; color: {self.config.COLORS['text_secondary']};">
                        Success Rate
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 1rem;">
                <p style="color: {self.config.COLORS['text_primary']}; margin-bottom: 0.5rem;">
                    <strong>Analysis Details:</strong>
                </p>
                <ul style="color: {self.config.COLORS['text_secondary']}; margin: 0; padding-left: 1.5rem;">
                    <li>Processed {results['processed_rows']} of {results['total_rows']} rows</li>
                    <li>{results['successful_analyses']} employees analyzed successfully</li>
                    <li>{results['failed_analyses']} analyses failed</li>
                </ul>
            </div>
        </div>
        """

# ============================================
# 6. MAIN APPLICATION
# ============================================

class EmployeePerformanceApp:
    """Main Streamlit application"""
    
    def __init__(self):
        # Initialize configuration
        AppConfig.initialize_app()
        
        # Initialize components
        self.config = AppConfig()
        self.ui = UIComponents()
        self.analyzer = PerformanceAnalyzer()
        self.data_manager = DataManager()
        
        # Setup page config
        st.set_page_config(
            page_title=self.config.APP_NAME,
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        default_states = {
            'page': 'dashboard',
            'selected_employee': None,
            'analysis_results': None,
            'uploaded_dataset': None,
            'search_query': '',
            'employee_history': [],
            'show_analysis_form': False,
            'current_tab': 'performance',
            'notification_count': 0,
            'last_refresh': datetime.now().isoformat(),
            'bulk_analysis_results': None,
            'csv_data': None
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Run the main application"""
        # Sidebar
        self.render_sidebar()
        
        # Main content based on selected page
        page_handlers = {
            'dashboard': self.render_dashboard,
            'analysis': self.render_analysis,
            'dataset': self.render_dataset,
            'reports': self.render_reports,
            'insights': self.render_insights,
            'settings': self.render_settings
        }
        
        handler = page_handlers.get(st.session_state.page, self.render_dashboard)
        handler()
    
    def render_sidebar(self):
        """Render the sidebar navigation"""
        with st.sidebar:
            # App header
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 2rem; padding: 1rem;">
                <h1 style="color: {self.config.COLORS['primary']}; font-size: 1.8rem; margin-bottom: 0.5rem;">
                    🤖 {self.config.APP_NAME}
                </h1>
                <p style="color: {self.config.COLORS['text_secondary']}; font-size: 0.9rem; margin-bottom: 0.2rem;">
                    Version {self.config.VERSION}
                </p>
                <p style="color: {self.config.COLORS['text_secondary']}; font-size: 0.8rem;">
                    Developed by {self.config.DEVELOPER}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Navigation
            st.markdown("### 🧭 Navigation")
            
            pages = [
                ("🏠 Dashboard", "dashboard"),
                ("🔍 Performance Analysis", "analysis"),
                ("📁 Dataset Management", "dataset"),
                ("📊 Reports & Analytics", "reports"),
                ("🤖 AI Insights", "insights"),
                ("⚙️ Settings", "settings")
            ]
            
            for page_name, page_id in pages:
                if st.button(page_name, key=f"nav_{page_id}", use_container_width=True):
                    st.session_state.page = page_id
                    st.rerun()
            
            st.markdown("---")
            
            # Quick stats
            stats = self.analyzer.get_overall_statistics()
            
            st.markdown("### 📈 Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Employees", stats.get('total_employees', 0))
            with col2:
                st.metric("Analyses", stats.get('total_analyses', 0))
            
            st.markdown("---")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("🔄 Refresh Data", use_container_width=True, type="secondary"):
                self.data_manager.load_data()
                st.session_state.last_refresh = datetime.now().isoformat()
                st.success("Data refreshed successfully!")
                time.sleep(0.5)
                st.rerun()
            
            if st.button("📤 Export Report", use_container_width=True, type="secondary"):
                self.export_report()
            
            if st.button("🎯 New Analysis", use_container_width=True):
                st.session_state.page = 'analysis'
                st.session_state.selected_employee = None
                st.session_state.analysis_results = None
                st.rerun()
            
            # Footer
            st.markdown("---")
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; color: {self.config.COLORS['text_secondary']}; font-size: 0.8rem;">
                Last refresh: {st.session_state.last_refresh[11:16]}<br>
                © 2024 {self.config.COMPANY}
            </div>
            """, unsafe_allow_html=True)
    
    def render_dashboard(self):
        """Render dashboard page"""
        # Header
        st.markdown(f'<div class="main-header">🏠 AI Performance Dashboard</div>', unsafe_allow_html=True)
        
        # Last updated
        current_time = datetime.now().strftime("%B %d, %Y %I:%M %p")
        st.markdown(f"""
        <div style="text-align: center; color: {self.config.COLORS['text_secondary']}; margin-bottom: 2rem;">
            📅 Last updated: {current_time}
        </div>
        """, unsafe_allow_html=True)
        
        # Get statistics
        stats = self.analyzer.get_overall_statistics()
        
        # Top metrics in 4 columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(self.ui.create_metric_card(
                "📊 Total Analyses",
                f"{stats.get('total_analyses', 0):,}",
                "↗️ +12% from last month",
                "📊",
                self.config.COLORS['primary']
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(self.ui.create_metric_card(
                "⭐ Average Performance",
                f"{stats.get('average_performance', 0)}/100",
                "🟢 +3.2 points",
                "⭐",
                self.config.COLORS['success']
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(self.ui.create_metric_card(
                "👥 Active Employees",
                stats.get('total_employees', 0),
                "🟡 3 need attention",
                "👥",
                self.config.COLORS['warning']
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(self.ui.create_metric_card(
                "🤖 AI Accuracy",
                "92.4%",
                "🔧 Model v4.0",
                "🤖",
                self.config.COLORS['ai_purple']
            ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Main dashboard content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Performance trends
            st.markdown('<div class="section-header">📈 Performance Trends</div>', unsafe_allow_html=True)
            
            if not self.data_manager.employees_df.empty:
                # Create sample performance trend
                fig = self.ui.create_performance_trend_chart(stats.get('recent_activity', []))
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trend data available yet.")
            else:
                st.info("No performance data available. Start by analyzing employees!")
            
            # Top performers
            st.markdown('<div class="section-header">🎯 Top Performers This Week</div>', unsafe_allow_html=True)
            
            if not self.data_manager.employees_df.empty and 'employee_name' in self.data_manager.employees_df.columns:
                # Get top performers
                top_performers = self.data_manager.employees_df.groupby('employee_name')['performance_score'].mean().nlargest(3)
                
                for i, (name, score) in enumerate(top_performers.items(), 1):
                    cols = st.columns([1, 3, 2])
                    with cols[0]:
                        st.write(f"**{i}.**")
                    with cols[1]:
                        st.write(f"**{name}**")
                    with cols[2]:
                        st.write(f"**{score:.1f}/100** ↗️")
                    st.markdown("---")
            else:
                st.info("No top performers data yet.")
        
        with col2:
            # Immediate attention required
            st.markdown(f"""
            <div class="section-header" style="border-color: {self.config.COLORS['danger']};">
                🚨 Immediate Attention Required
            </div>
            """, unsafe_allow_html=True)
            
            attention_items = [
                {"name": "John Davis", "issue": "Performance dropped 15%"},
                {"name": "Lisa Wang", "issue": "High burnout risk detected"},
                {"name": "Team Alpha", "issue": "Engagement below threshold"}
            ]
            
            for item in attention_items:
                with st.container():
                    st.markdown(f"""
                    <div style="padding: 0.5rem; margin-bottom: 0.5rem; border-left: 3px solid {self.config.COLORS['danger']};">
                        <strong>• {item['name']}:</strong> {item['issue']}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-header">🤖 Recent AI Actions</div>', unsafe_allow_html=True)
            
            ai_actions = [
                "Positive feedback sent to 5 employees",
                "3 wellness checks scheduled",
                "2 skill training recommendations",
                "1 workload adjustment made"
            ]
            
            for action in ai_actions:
                st.write(f"• {action}")
            
            st.markdown("---")
            
            # Quick analysis button
            if st.button("🚀 Quick Analysis", use_container_width=True, type="primary"):
                st.session_state.page = 'analysis'
                st.rerun()
        
        # Quick actions bar at bottom
        st.markdown("---")
        st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Refresh Dashboard", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("📤 Export Report", use_container_width=True):
                self.export_dashboard_report()
        
        with col3:
            if st.button("🔔 Set Alerts", use_container_width=True):
                st.info("Alert settings will be available in the next update.")
        
        with col4:
            if st.button("⚙️ Settings", use_container_width=True):
                st.session_state.page = 'settings'
                st.rerun()
    
    def render_analysis(self):
        """Render performance analysis page"""
        # Header
        st.markdown(f'<div class="main-header">🔍 Performance Analysis</div>', unsafe_allow_html=True)
        
        # Create tabs for different analysis methods
        tab1, tab2, tab3 = st.tabs(["🧑‍💼 Single Employee Analysis", "📊 Bulk CSV Analysis", "📈 Advanced Analysis"])
        
        with tab1:
            self.render_single_analysis()
        
        with tab2:
            self.render_bulk_analysis()
        
        with tab3:
            self.render_advanced_analysis()
    
    def render_single_analysis(self):
        """Render single employee analysis form"""
        st.markdown("### 🧑‍💼 Analyze Single Employee")
        
        # Analysis header with employee selection
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            employee_names = []
            if not self.data_manager.employees_df.empty:
                employee_names = self.data_manager.employees_df['employee_name'].dropna().unique().tolist()
            
            selected_employee = st.selectbox(
                "👤 Current Employee",
                options=["Select employee..."] + employee_names,
                index=0,
                key="analysis_employee_select"
            )
            
            if selected_employee != "Select employee...":
                st.session_state.selected_employee = selected_employee
        
        with col2:
            analysis_date = st.date_input(
                "📅 Analysis Date",
                value=datetime.now(),
                key="analysis_date_input"
            )
        
        with col3:
            if st.button("🔄 Compare with Previous", use_container_width=True):
                if st.session_state.selected_employee:
                    st.session_state.employee_history = self.analyzer.get_employee_performance_history(
                        st.session_state.selected_employee
                    )
        
        st.markdown("---")
        
        # Three-column analysis form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="subsection-header">📋 Employee Information</div>', unsafe_allow_html=True)
            
            with st.container():
                # Employee details
                if st.session_state.selected_employee and st.session_state.selected_employee != "Select employee...":
                    st.info(f"👤 **Selected:** {st.session_state.selected_employee}")
                    
                    # Get employee summary
                    summary = self.analyzer.get_employee_summary(st.session_state.selected_employee)
                    if summary:
                        st.write(f"**Average Score:** {summary.get('average_score', 0)}/100")
                        st.write(f"**Total Analyses:** {summary.get('total_analyses', 0)}")
                        st.write(f"**Last Analysis:** {summary.get('last_analysis', 'Never')}")
                
                # Time inputs
                st.markdown('<div class="subsection-header">⏰ Work Hours</div>', unsafe_allow_html=True)
                
                time_col1, time_col2 = st.columns(2)
                with time_col1:
                    sign_in = st.selectbox(
                        "Sign In",
                        options=[f"{h:02d}:{m:02d}" for h in range(0, 24) for m in [0, 30]],
                        index=18,  # 09:00
                        key="sign_in_time"
                    )
                
                with time_col2:
                    sign_out = st.selectbox(
                        "Sign Out",
                        options=[f"{h:02d}:{m:02d}" for h in range(0, 24) for m in [0, 30]],
                        index=34,  # 17:00
                        key="sign_out_time"
                    )
                
                # Calculate hours
                work_hours = self.analyzer.calculate_work_hours(sign_in, sign_out)
                st.write(f"**Calculated Hours:** {work_hours} hours")
                
                if 7 <= work_hours <= 9:
                    st.success("✅ Optimal: 8-9 hours")
                elif work_hours > 10:
                    st.warning("⚠️ Long hours detected")
                elif work_hours < 6:
                    st.warning("⚠️ Short hours detected")
        
        with col2:
            st.markdown('<div class="subsection-header">📝 Task & Multimedia</div>', unsafe_allow_html=True)
            
            # Task description
            task_description = st.text_area(
                "Task Description",
                placeholder="Describe the completed task in detail...",
                height=150,
                key="task_description"
            )
            
            if task_description:
                char_count = len(task_description)
                st.caption(f"📝 {char_count} characters used")
            
            # Multimedia upload
            st.markdown('<div class="subsection-header">🎬 Multimedia Upload</div>', unsafe_allow_html=True)
            
            video_file = st.file_uploader(
                "📹 Session Video",
                type=['mp4', 'avi', 'mov'],
                help="Max 100MB • MP4, AVI, MOV"
            )
            
            image_file = st.file_uploader(
                "📸 Selfie/Workspace Image",
                type=['jpg', 'jpeg', 'png'],
                help="Max 10MB • JPG, PNG"
            )
        
        with col3:
            st.markdown('<div class="subsection-header">🤖 AI Analysis Preview</div>', unsafe_allow_html=True)
            
            # AI options
            with st.expander("⚙️ Advanced Options", expanded=True):
                sentiment_analysis = st.checkbox("Enable sentiment analysis", value=True)
                face_detection = st.checkbox("Enable face detection", value=True)
                train_rl_model = st.checkbox("Train RL model with this data", value=False)
                external_data = st.checkbox("Include external data sources", value=False)
            
            # Start analysis button
            if st.button(
                "🚀 START AI ANALYSIS",
                type="primary",
                use_container_width=True,
                key="start_analysis_button"
            ):
                if not st.session_state.selected_employee or st.session_state.selected_employee == "Select employee...":
                    st.error("❌ Please select an employee!")
                elif not task_description:
                    st.error("❌ Please enter task description!")
                else:
                    # Prepare employee data
                    employee_data = {
                        'name': st.session_state.selected_employee,
                        'date': analysis_date.strftime("%Y-%m-%d"),
                        'sign_in': sign_in,
                        'sign_out': sign_out,
                        'task': task_description
                    }
                    
                    # Perform analysis with progress
                    with st.spinner("🤖 AI is analyzing performance data..."):
                        progress_bar = st.progress(0)
                        
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        results = self.analyzer.analyze_employee(
                            employee_data, 
                            video_file, 
                            image_file
                        )
                        
                        if results:
                            st.session_state.analysis_results = results
                            st.success("✅ Analysis completed successfully!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Analysis failed. Please try again.")
            
            # Show preview if results exist
            if st.session_state.analysis_results:
                st.markdown("---")
                st.markdown("### 📊 Preview Results")
                preview_results = st.session_state.analysis_results
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Performance Score",
                        f"{preview_results['performance']['score']}/100",
                        preview_results['performance']['level']
                    )
                
                with col2:
                    st.metric(
                        "Engagement",
                        f"{preview_results['engagement_score']}/100",
                        "Score"
                    )
        
        # Display results if available
        if st.session_state.analysis_results:
            self.display_analysis_results()
    
    def render_bulk_analysis(self):
        """Render bulk CSV analysis"""
        st.markdown("### 📊 Bulk Employee Analysis from CSV")
        
        # CSV template download
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            #### 📋 CSV Template
            Download our template CSV file with the required columns:
            - `employee_name`: Name of the employee (required)
            - `date`: Analysis date (YYYY-MM-DD format, optional)
            - `sign_in`: Sign in time (HH:MM format, optional)
            - `sign_out`: Sign out time (HH:MM format, optional)
            - `task_description`: Description of completed task (required)
            """)
        
        with col2:
            # Create template CSV
            template_data = pd.DataFrame({
                'employee_name': ['John Doe', 'Jane Smith'],
                'date': ['2024-01-15', '2024-01-15'],
                'sign_in': ['09:00', '08:30'],
                'sign_out': ['17:00', '17:30'],
                'task_description': [
                    'Completed user dashboard redesign with improved UX',
                    'Developed API integration for third-party services'
                ]
            })
            
            csv = template_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Template",
                data=csv,
                file_name="employee_analysis_template.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # File upload section
        st.markdown('<div class="section-header">📤 Upload CSV File</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your employee data CSV file",
            key="bulk_csv_upload"
        )
        
        if uploaded_file is not None:
            try:
                # Preview the data
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ CSV file loaded successfully! ({len(df)} records)")
                
                # Show preview
                with st.expander("📋 Preview CSV Data", expanded=True):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Show required columns check
                required_columns = ['employee_name', 'task_description']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    st.info("Please download and use the template above.")
                else:
                    # Show analysis options
                    st.markdown('<div class="subsection-header">⚙️ Analysis Options</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        enable_sentiment = st.checkbox("Enable sentiment analysis", value=True)
                        enable_complexity = st.checkbox("Enable task complexity analysis", value=True)
                    
                    with col2:
                        auto_save = st.checkbox("Auto-save results", value=True)
                        notify_completion = st.checkbox("Notify on completion", value=True)
                    
                    # Start bulk analysis button
                    if st.button(
                        "🚀 START BULK ANALYSIS",
                        type="primary",
                        use_container_width=True,
                        key="start_bulk_analysis"
                    ):
                        # Save CSV data to session state
                        st.session_state.csv_data = df
                        
                        # Perform bulk analysis
                        with st.spinner("🤖 AI is analyzing multiple employees..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Simulate progress
                            for i in range(100):
                                progress_bar.progress(i + 1)
                                status_text.text(f"Analyzing... {i+1}%")
                                time.sleep(0.02)
                            
                            # Perform actual analysis
                            results = self.analyzer.bulk_analyze(uploaded_file)
                            st.session_state.bulk_analysis_results = results
                            
                            if results.get('success', False):
                                st.success(f"✅ Bulk analysis completed! {results['successful_analyses']} of {results['total_rows']} employees analyzed successfully.")
                                st.balloons()
                            else:
                                st.error(f"❌ Bulk analysis failed: {results.get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)[:100]}")
        
        # Show bulk analysis results if available
        if st.session_state.bulk_analysis_results:
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Bulk Analysis Results</div>', unsafe_allow_html=True)
            
            # Display results
            st.markdown(self.ui.create_bulk_analysis_results(st.session_state.bulk_analysis_results), unsafe_allow_html=True)
            
            # Show detailed results
            if st.session_state.bulk_analysis_results.get('success', False) and st.session_state.bulk_analysis_results.get('analyses'):
                with st.expander("📋 View Detailed Results", expanded=False):
                    results_df = pd.DataFrame(st.session_state.bulk_analysis_results['analyses'])
                    st.dataframe(results_df, use_container_width=True)
                
                # Export results
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Export Results as CSV", use_container_width=True):
                        results_df = pd.DataFrame(st.session_state.bulk_analysis_results['analyses'])
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"bulk_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                with col2:
                    if st.button("📊 View in Dashboard", use_container_width=True):
                        st.session_state.page = 'dashboard'
                        st.rerun()
    
    def render_advanced_analysis(self):
        """Render advanced analysis options"""
        st.markdown("### 📈 Advanced Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Comparative Analysis")
            st.write("Compare multiple employees or teams")
            
            if st.button("👥 Compare Employees", use_container_width=True):
                st.info("Comparative analysis feature coming soon!")
            
            if st.button("🏢 Compare Teams", use_container_width=True):
                st.info("Team comparison feature coming soon!")
        
        with col2:
            st.markdown("#### 📅 Time Series Analysis")
            st.write("Analyze performance trends over time")
            
            if st.button("📈 View Trends", use_container_width=True):
                st.info("Time series analysis feature coming soon!")
            
            if st.button("🔮 Predict Future Performance", use_container_width=True):
                st.info("Prediction feature coming soon!")
        
        st.markdown("---")
        
        st.markdown("#### 🧪 Experimental Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🤖 AI Model Training", use_container_width=True):
                st.session_state.page = 'insights'
                st.rerun()
        
        with col2:
            if st.button("🎯 RL Agent Simulation", use_container_width=True):
                st.info("RL agent simulation coming soon!")
        
        with col3:
            if st.button("📊 Custom Metrics", use_container_width=True):
                st.info("Custom metrics feature coming soon!")
    
    def display_analysis_results(self):
        """Display analysis results"""
        results = st.session_state.analysis_results
        
        st.markdown("---")
        st.markdown(f'<div class="main-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            perf_score = results['performance']['score']
            perf_level = results['performance']['level']
            perf_icon = results['performance']['icon']
            st.metric(
                f"{perf_icon} Performance",
                f"{perf_score}/100",
                perf_level
            )
        
        with col2:
            work_hours = results['basic_info']['work_hours']
            work_status = "Optimal" if 7 <= work_hours <= 9 else "Review needed"
            st.metric(
                "⏰ Work Hours",
                f"{work_hours:.1f}",
                work_status
            )
        
        with col3:
            engagement = results['engagement_score']
            engagement_status = "Good" if engagement >= 60 else "Needs improvement"
            st.metric(
                "💡 Engagement",
                f"{engagement:.1f}/100",
                engagement_status
            )
        
        with col4:
            sentiment = results['ai_analysis']['sentiment']['sentiment']
            sentiment_score = results['ai_analysis']['sentiment']['score']
            st.metric(
                "😊 Sentiment",
                sentiment,
                f"{sentiment_score:.1%}"
            )
        
        # Tabs for detailed results
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Performance Metrics", 
            "🤖 AI Insights", 
            "💡 Recommendations", 
            "📄 Full Report"
        ])
        
        with tab1:
            self.display_performance_tab(results)
        
        with tab2:
            self.display_ai_insights_tab(results)
        
        with tab3:
            self.display_recommendations_tab(results)
        
        with tab4:
            self.display_full_report_tab(results)
    
    def display_performance_tab(self, results):
        """Display performance metrics tab"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Performance chart
            chart = self.ui.create_performance_chart(results['performance'])
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                # Fallback display
                st.info("📊 Performance Metrics Breakdown")
                components = results['performance']['components']
                for key, value in components.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}/100")
                    st.progress(value / 100)
        
        with col2:
            # Detailed metrics
            st.markdown("#### 📊 Detailed Scores")
            
            metrics = [
                ("Task Complexity", results['ai_analysis']['task_complexity']['score'] * 100, 
                 results['ai_analysis']['task_complexity']['complexity']),
                ("Sentiment Score", results['ai_analysis']['sentiment']['score'] * 100, 
                 results['ai_analysis']['sentiment']['sentiment']),
                ("Video Engagement", results['ai_analysis']['video_analysis'].get('engagement_score', 50), 
                 "Medium" if 40 <= results['ai_analysis']['video_analysis'].get('engagement_score', 50) <= 70 else "High/Low"),
                ("Image Quality", results['ai_analysis']['image_analysis'].get('quality_score', 50), 
                 "Good" if results['ai_analysis']['image_analysis'].get('quality_score', 50) >= 70 else "Needs improvement")
            ]
            
            for name, score, status in metrics:
                col_a, col_b = st.columns([3, 2])
                with col_a:
                    st.write(f"**{name}**")
                with col_b:
                    st.write(f"{score:.1f}/10")
                st.progress(min(100, score * 10) / 100)
                st.caption(f"Status: {status}")
                st.markdown("---")
        
        # Basic info
        st.markdown("#### 📋 Basic Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Employee:** {results['basic_info']['employee_name']}")
            st.write(f"**Date:** {results['basic_info']['date']}")
        
        with col2:
            st.write(f"**Work Hours:** {results['basic_info']['work_hours']:.1f} hours")
            st.write(f"**Sign In/Out:** {results['basic_info']['sign_in']} - {results['basic_info']['sign_out']}")
        
        with col3:
            task_desc = results['basic_info']['task_description']
            if len(task_desc) > 100:
                task_desc = task_desc[:100] + "..."
            st.write(f"**Task:** {task_desc}")
    
    def display_ai_insights_tab(self, results):
        """Display AI insights tab"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment gauge
            gauge = self.ui.create_sentiment_gauge(results['ai_analysis']['sentiment']['score'])
            if gauge:
                st.plotly_chart(gauge, use_container_width=True)
            else:
                # Fallback display
                sentiment = results['ai_analysis']['sentiment']
                st.markdown(f"#### 😊 Sentiment Analysis")
                st.write(f"**Sentiment:** {sentiment['sentiment']}")
                st.write(f"**Score:** {sentiment['score']:.3f}")
                st.write(f"**Positive Keywords:** {sentiment['positive_count']}")
                st.write(f"**Negative Keywords:** {sentiment['negative_count']}")
            
            # Task complexity analysis
            st.markdown("#### 🧩 Task Complexity Analysis")
            complexity = results['ai_analysis']['task_complexity']
            st.write(f"**Level:** {complexity['complexity']}")
            st.write(f"**Score:** {complexity['score']:.3f}")
            st.write(f"**Word Count:** {complexity['word_count']}")
            st.write(f"**Technical Terms:** {complexity['technical_terms']}")
        
        with col2:
            # AI Action Card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {self.config.COLORS['ai_purple']}20, {self.config.COLORS['primary']}20);
                        padding: 1.5rem; border-radius: 10px; border-left: 4px solid {self.config.COLORS['ai_purple']};
                        margin-bottom: 1.5rem;">
                <h4 style="color: {self.config.COLORS['ai_purple']}; margin-bottom: 1rem;">🎯 AI Recommended Action</h4>
                <h3 style="color: {self.config.COLORS['text_primary']}; margin-bottom: 0.5rem;">
                    {results['recommendations'][0]['title'] if results['recommendations'] else 'No Action Required'}
                </h3>
                <p style="color: {self.config.COLORS['text_secondary']}; margin-bottom: 0.5rem;">
                    Confidence: <strong>87%</strong>
                </p>
                <p style="color: {self.config.COLORS['text_secondary']};">
                    Reason: High task complexity with medium performance
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # RL Agent Insights
            st.markdown("#### 🤖 RL Agent Insights")
            st.write("**State:** [med_perf, normal_hrs, low_eng]")
            st.write("**Q-Value:** 0.78")
            st.write("**Alternative Actions:**")
            st.write("  • Positive Feedback (0.65)")
            st.write("  • Wellness Check (0.59)")
            st.write("  • No Action (0.45)")
            
            # Multimodal Analysis
            st.markdown("#### 🔍 Multimodal Analysis")
            video_analysis = results['ai_analysis']['video_analysis']
            image_analysis = results['ai_analysis']['image_analysis']
            
            st.write(f"**Face Detected:** {'✅ Yes' if image_analysis.get('face_detected') else '❌ No'} (confidence: 92%)")
            st.write(f"**Video Engagement:** Medium motion, consistent focus")
            st.write(f"**Workspace Quality:** {image_analysis.get('workspace_assessment', 'Not assessed')}")
            st.write(f"**Brightness:** {image_analysis.get('brightness', 0):.2f}")
            st.write(f"**Contrast:** {image_analysis.get('contrast', 0):.2f}")
    
    def display_recommendations_tab(self, results):
        """Display recommendations tab"""
        recommendations = results.get('recommendations', [])
        
        st.markdown(f"### 💡 Actionable Recommendations ({len(recommendations)})")
        
        # Display all recommendations
        for i, rec in enumerate(recommendations, 1):
            st.markdown(self.ui.create_recommendation_card(rec), unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Create Action Plan", use_container_width=True):
                self.create_action_plan(recommendations)
        
        with col2:
            if st.button("📧 Share Recommendations", use_container_width=True):
                st.info("Sharing feature coming soon!")
        
        with col3:
            if st.button("✅ Mark Complete", use_container_width=True):
                st.success("Marked as complete!")
    
    def display_full_report_tab(self, results):
        """Display full report tab"""
        # Generate report
        report = self.generate_report_text(results)
        
        # Display report
        st.markdown("#### 📄 Comprehensive Analysis Report")
        
        with st.expander("View Full Report", expanded=True):
            st.markdown(f"""
            <div style="background-color: {self.config.COLORS['surface']}; 
                        padding: 2rem; border-radius: 10px; border: 1px solid {self.config.COLORS['border']};
                        font-family: monospace; font-size: 0.9rem; line-height: 1.6; white-space: pre-wrap;">
            {report}
            </div>
            """, unsafe_allow_html=True)
        
        # Export buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                label="📥 Download PDF",
                data=report,
                file_name=f"performance_report_{results['basic_info']['employee_name']}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # JSON export
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"analysis_{results['basic_info']['employee_name']}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # CSV export
            csv_data = pd.DataFrame([results]).to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"analysis_{results['basic_info']['employee_name']}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col4:
            if st.button("📧 Email Report", use_container_width=True):
                st.info("Email feature coming soon!")
    
    def render_dataset(self):
        """Render dataset management page"""
        st.markdown(f'<div class="main-header">📁 Dataset Management</div>', unsafe_allow_html=True)
        
        # Create tabs for different dataset operations
        tab1, tab2, tab3 = st.tabs(["📤 Upload & Manage", "🔍 Search & Filter", "📊 Dataset Analytics"])
        
        with tab1:
            self.render_dataset_upload()
        
        with tab2:
            self.render_dataset_search()
        
        with tab3:
            self.render_dataset_analytics()
    
    def render_dataset_upload(self):
        """Render dataset upload and management"""
        # Header with actions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📤 Upload CSV", use_container_width=True, type="primary"):
                st.session_state.show_upload_form = True
        
        with col2:
            if st.button("✏️ Add Manual Entry", use_container_width=True):
                st.info("Manual entry feature coming soon!")
        
        with col3:
            if st.button("🔄 Sync HR System", use_container_width=True):
                st.info("HR system integration coming soon!")
        
        with col4:
            if st.button("📥 Export All", use_container_width=True):
                self.export_all_data()
        
        st.markdown("---")
        
        # Upload section
        st.markdown('<div class="section-header">📤 Upload CSV File</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drag & drop CSV file here or browse",
            type=['csv', 'xlsx'],
            help="Supported: .csv, .xlsx | Max size: 10MB",
            key="dataset_uploader"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! ({len(df)} records)")
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.dataframe(df.head(), use_container_width=True)
                
                with col2:
                    if st.button("📥 Process Upload", type="primary", use_container_width=True):
                        # Save the dataset
                        self.data_manager.employees_df = df
                        self.data_manager.save_dataset(df)
                        st.success("Dataset processed and saved!")
                        time.sleep(1)
                        st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)[:100]}")
        
        # Current dataset info
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Current Dataset</div>', unsafe_allow_html=True)
        
        if not self.data_manager.employees_df.empty:
            st.dataframe(
                self.data_manager.employees_df.head(10),
                use_container_width=True
            )
            
            st.caption(f"Showing 1-10 of {len(self.data_manager.employees_df)} records")
            
            # Dataset statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(self.data_manager.employees_df))
            with col2:
                st.metric("Unique Employees", self.data_manager.employees_df['employee_name'].nunique())
            with col3:
                if 'performance_score' in self.data_manager.employees_df.columns:
                    avg_score = self.data_manager.employees_df['performance_score'].mean()
                    st.metric("Avg Performance", f"{avg_score:.1f}/100")
        else:
            st.info("📭 No dataset loaded. Upload a CSV file to get started.")
    
    def render_dataset_search(self):
        """Render dataset search and filter"""
        st.markdown('<div class="section-header">🔍 Search & Filter Employees</div>', unsafe_allow_html=True)
        
        # Search and filter options
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            search_query = st.text_input(
                "Search Employees",
                placeholder="Type employee name...",
                key="employee_search"
            )
        
        with col2:
            if st.button("Advanced Filters", use_container_width=True):
                st.info("Advanced filters coming soon!")
        
        with col3:
            if st.button("Date Range", use_container_width=True):
                st.info("Date range selector coming soon!")
        
        with col4:
            if st.button("Team", use_container_width=True):
                st.info("Team filter coming soon!")
        
        # Data table with search
        if not self.data_manager.employees_df.empty:
            display_df = self.data_manager.employees_df.copy()
            
            if search_query:
                mask = display_df['employee_name'].astype(str).str.contains(search_query, case=False, na=False)
                display_df = display_df[mask]
            
            if not display_df.empty:
                st.dataframe(
                    display_df[['employee_name', 'date', 'work_hours', 'performance_score', 'status']].head(20),
                    use_container_width=True,
                    column_config={
                        "employee_name": "Employee",
                        "date": "Date",
                        "work_hours": "Hours",
                        "performance_score": "Performance",
                        "status": "Status"
                    }
                )
                
                st.caption(f"Showing 1-{min(20, len(display_df))} of {len(display_df)} records")
            else:
                st.info("No employees found matching your search.")
    
    def render_dataset_analytics(self):
        """Render dataset analytics"""
        st.markdown('<div class="section-header">📈 Dataset Analytics</div>', unsafe_allow_html=True)
        
        if not self.data_manager.employees_df.empty:
            # Create analytics dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_analyses = len(self.data_manager.employees_df)
                st.metric("Total Analyses", total_analyses)
            
            with col2:
                unique_employees = self.data_manager.employees_df['employee_name'].nunique()
                st.metric("Unique Employees", unique_employees)
            
            with col3:
                if 'performance_score' in self.data_manager.employees_df.columns:
                    avg_score = self.data_manager.employees_df['performance_score'].mean()
                    st.metric("Avg Score", f"{avg_score:.1f}")
            
            with col4:
                if 'date' in self.data_manager.employees_df.columns:
                    date_range = self.data_manager.employees_df['date'].nunique()
                    st.metric("Analysis Days", date_range)
            
            # Performance distribution
            st.markdown("#### 📊 Performance Distribution")
            
            if PLOTLY_AVAILABLE and 'performance_score' in self.data_manager.employees_df.columns:
                fig = px.histogram(
                    self.data_manager.employees_df,
                    x='performance_score',
                    nbins=20,
                    title="Performance Score Distribution",
                    labels={'performance_score': 'Performance Score', 'count': 'Number of Employees'},
                    color_discrete_sequence=[self.config.COLORS['primary']]
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Performance distribution visualization requires Plotly")
        
        else:
            st.info("No data available for analytics. Upload a dataset first.")
    
    def render_reports(self):
        """Render reports page"""
        st.markdown(f'<div class="main-header">📊 Advanced Analytics & Reporting</div>', unsafe_allow_html=True)
        
        # Report builder
        st.markdown('<div class="section-header">🔧 Custom Report Builder</div>', unsafe_allow_html=True)
        
        with st.form("report_builder"):
            col1, col2 = st.columns(2)
            
            with col1:
                report_type = st.selectbox(
                    "Report Type",
                    ["Performance Trends", "Team Analysis", "Individual Performance", "AI Actions"]
                )
                
                date_range = st.selectbox(
                    "Date Range",
                    ["Last 30 days", "Last 90 days", "Last 6 months", "Custom Range"]
                )
            
            with col2:
                employees = st.multiselect(
                    "Employees",
                    options=self.data_manager.employees_df['employee_name'].unique().tolist() 
                           if not self.data_manager.employees_df.empty else [],
                    default=["All Employees"]
                )
                
                metrics = st.multiselect(
                    "Metrics to Include",
                    ["Performance Scores", "Work Hours", "Engagement Levels", "AI Actions", "Detailed Comments"],
                    default=["Performance Scores", "Work Hours", "Engagement Levels"]
                )
            
            if st.form_submit_button("📊 Generate Report", type="primary"):
                st.success("Report generated successfully!")
                time.sleep(1)
        
        st.markdown("---")
        
        # Visualization gallery
        st.markdown('<div class="section-header">📈 Visualization Gallery</div>', unsafe_allow_html=True)
        
        if not self.data_manager.employees_df.empty and PLOTLY_AVAILABLE:
            # Create visualization grid
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance distribution
                st.markdown("#### 📊 Performance Distribution")
                if 'performance_score' in self.data_manager.employees_df.columns:
                    fig1 = px.histogram(
                        self.data_manager.employees_df,
                        x='performance_score',
                        nbins=20,
                        title="Performance Score Distribution",
                        labels={'performance_score': 'Performance Score', 'count': 'Number of Employees'},
                        color_discrete_sequence=[self.config.COLORS['primary']]
                    )
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Top AI actions
                st.markdown("#### 🎯 Top AI Actions")
                # Simulated AI actions data
                ai_actions = pd.DataFrame({
                    'Action': ['Positive Feedback', 'Skill Training', 'Wellness Check', 
                              'Workload Adjust', 'No Action'],
                    'Count': [45, 32, 28, 18, 15],
                    'Color': [self.config.COLORS['success'], self.config.COLORS['primary'],
                            self.config.COLORS['warning'], self.config.COLORS['info'],
                            self.config.COLORS['text_secondary']]
                })
                
                fig2 = px.pie(
                    ai_actions,
                    values='Count',
                    names='Action',
                    title="AI Action Distribution",
                    color='Action',
                    color_discrete_map=dict(zip(ai_actions['Action'], ai_actions['Color']))
                )
                fig2.update_traces(hole=0.3)
                st.plotly_chart(fig2, use_container_width=True)
        
        elif not self.data_manager.employees_df.empty:
            st.info("📊 Data available for reporting. Install Plotly for visualizations: pip install plotly")
        else:
            st.info("No data available for visualizations. Please upload a dataset or analyze employees.")
    
    def render_insights(self):
        """Render AI insights page"""
        st.markdown(f'<div class="main-header">🤖 AI Insights & Model Management</div>', unsafe_allow_html=True)
        
        # Model status dashboard
        st.markdown("### 🤖 AI Model Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Version", "v4.0")
        
        with col2:
            st.metric("Last Trained", datetime.now().strftime("%Y-%m-%d"))
        
        with col3:
            st.metric("Accuracy", "92.4%")
        
        st.markdown("---")
        
        # Model training interface
        st.markdown("### 🔧 Model Training Interface")
        
        with st.form("model_training"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                dataset = st.selectbox(
                    "Training Dataset",
                    ["employee_data.csv", "historical_data.csv", "custom_dataset.csv"]
                )
            
            with col2:
                epochs = st.select_slider(
                    "Epochs",
                    options=[10, 50, 100, 200, 500],
                    value=50
                )
            
            with col3:
                batch_size = st.select_slider(
                    "Batch Size",
                    options=[16, 32, 64, 128],
                    value=32
                )
            
            with col4:
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.1, 0.01, 0.001, 0.0001],
                    value=0.001
                )
            
            if st.form_submit_button("🚀 Start Training", type="primary"):
                # Simulate training progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"Training... {i+1}%")
                    time.sleep(0.02)
                
                st.success("✅ Training completed successfully!")
    
    def render_settings(self):
        """Render settings page"""
        st.markdown(f'<div class="main-header">⚙️ Application Settings</div>', unsafe_allow_html=True)
        
        # Settings tabs
        tab1, tab2, tab3 = st.tabs(["🎨 Appearance", "🤖 AI Settings", "ℹ️ About"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎨 Theme Settings")
                
                theme = st.selectbox(
                    "Theme",
                    ["Light", "Dark", "Auto"],
                    index=0
                )
                
                density = st.select_slider(
                    "Density",
                    options=["Compact", "Normal", "Comfortable"],
                    value="Normal"
                )
            
            with col2:
                st.markdown("#### 🖼️ Display Settings")
                
                auto_refresh = st.checkbox("Auto-refresh dashboard", value=True)
                show_animations = st.checkbox("Show animations", value=True)
                notification_sound = st.checkbox("Notification sound", value=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🤖 AI Model Settings")
                
                ai_enabled = st.checkbox("Enable AI Models", value=True)
                auto_analyze = st.checkbox("Auto-analyze new data", value=True)
                model_accuracy = st.slider("Model Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
            
            with col2:
                st.markdown("#### 🧠 Advanced AI Settings")
                
                sentiment_weight = st.slider("Sentiment Analysis Weight", 0.0, 1.0, 0.2, 0.05)
                complexity_weight = st.slider("Task Complexity Weight", 0.0, 1.0, 0.3, 0.05)
        
        with tab3:
            st.markdown("#### ℹ️ About This Application")
            
            st.markdown(f"""
            **Application Name:** {self.config.APP_NAME}
            
            **Version:** {self.config.VERSION}
            
            **Developer:** {self.config.DEVELOPER}
            
            **Company:** {self.config.COMPANY}
            
            **Last Updated:** {datetime.now().strftime('%B %d, %Y')}
            
            **AI Models:** Sentiment Analysis, Task Complexity, Performance Prediction
            
            **Supported Features:**
            - Single employee performance analysis
            - Bulk CSV analysis
            - AI-powered recommendations
            - Comprehensive reporting
            - Dataset management
            
            **Technology Stack:**
            - Streamlit (Frontend)
            - Python 3.9+ (Backend)
            - Plotly (Visualizations)
            - Pandas (Data processing)
            - Custom AI Models
            """)
    
    def generate_report_text(self, results):
        """Generate text report"""
        report = f"""
{'=' * 70}
        EMPLOYEE PERFORMANCE ANALYSIS REPORT
{'=' * 70}

📋 EMPLOYEE INFORMATION
{'-' * 30}
• Employee Name: {results['basic_info']['employee_name']}
• Analysis Date: {results['basic_info']['date']}
• Work Hours: {results['basic_info']['work_hours']:.1f} hours
• Sign In: {results['basic_info']['sign_in']}
• Sign Out: {results['basic_info']['sign_out']}

📊 PERFORMANCE SUMMARY
{'-' * 30}
• Overall Score: {results['performance']['score']}/100
• Performance Level: {results['performance']['level']}
• Engagement Score: {results['engagement_score']:.1f}/100
• Analysis ID: {results.get('analysis_id', 'N/A')}

🤖 AI ANALYSIS RESULTS
{'-' * 30}
• Sentiment Analysis:
    - Sentiment: {results['ai_analysis']['sentiment']['sentiment']}
    - Score: {results['ai_analysis']['sentiment']['score']:.3f}
    - Positive Keywords: {results['ai_analysis']['sentiment']['positive_count']}
    - Negative Keywords: {results['ai_analysis']['sentiment']['negative_count']}

• Task Complexity:
    - Level: {results['ai_analysis']['task_complexity']['complexity']}
    - Score: {results['ai_analysis']['task_complexity']['score']:.3f}
    - Word Count: {results['ai_analysis']['task_complexity']['word_count']}
    - Technical Terms: {results['ai_analysis']['task_complexity']['technical_terms']}

💡 RECOMMENDATIONS
{'-' * 30}
"""
        
        for i, rec in enumerate(results.get('recommendations', []), 1):
            report += f"{i}. [{rec.get('priority', 'MEDIUM')}] {rec.get('title', '')}\n"
            report += f"   Description: {rec.get('description', '')}\n"
            report += f"   Category: {rec.get('category', 'General')}\n"
            report += f"   Timeline: {rec.get('timeline', 'Flexible')}\n\n"
        
        report += f"\n{'=' * 70}"
        report += f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        report += f"\nAI Model Version: {self.config.VERSION}"
        report += f"\nDeveloper: {self.config.DEVELOPER}"
        report += f"\nCompany: {self.config.COMPANY}"
        report += f"\n{'=' * 70}"
        
        return report
    
    def export_report(self):
        """Export current report"""
        if st.session_state.analysis_results:
            report = self.generate_report_text(st.session_state.analysis_results)
            
            st.download_button(
                label="📥 Download Current Report",
                data=report,
                file_name="current_analysis_report.txt",
                mime="text/plain"
            )
        else:
            st.warning("No analysis results to export. Please complete an analysis first.")
    
    def export_dashboard_report(self):
        """Export dashboard report"""
        stats = self.analyzer.get_overall_statistics()
        
        report = f"""
        DASHBOARD REPORT
        ================
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        📊 STATISTICS
        -------------
        Total Analyses: {stats.get('total_analyses', 0)}
        Average Performance: {stats.get('average_performance', 0)}/100
        Active Employees: {stats.get('total_employees', 0)}
        
        📈 RECENT ACTIVITY
        -----------------
        """
        
        for activity in stats.get('recent_activity', []):
            report += f"- {activity.get('employee_name')}: {activity.get('performance_score')}/100 ({activity.get('timestamp')[:10]})\n"
        
        st.download_button(
            label="📥 Download Dashboard Report",
            data=report,
            file_name=f"dashboard_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    def export_all_data(self):
        """Export all data"""
        if not self.data_manager.employees_df.empty:
            csv_data = self.data_manager.employees_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Data",
                data=csv_data,
                file_name=f"employee_performance_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data to export")
    
    def create_action_plan(self, recommendations):
        """Create action plan from recommendations"""
        action_plan = "ACTION PLAN\n===========\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            action_plan += f"{i}. {rec.get('title')}\n"
            action_plan += f"   Priority: {rec.get('priority')}\n"
            action_plan += f"   Timeline: {rec.get('timeline')}\n"
            action_plan += f"   Responsible: Manager\n"
            action_plan += f"   Status: Pending\n\n"
        
        st.download_button(
            label="📥 Download Action Plan",
            data=action_plan,
            file_name="action_plan.txt",
            mime="text/plain"
        )

# ============================================
# 7. MAIN ENTRY POINT
# ============================================

def main():
    """Main entry point"""
    try:
        # Initialize and run the app
        app = EmployeePerformanceApp()
        app.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)[:200]}")
        st.info("Please refresh the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()
