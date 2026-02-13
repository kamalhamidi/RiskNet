"""
🎓 Student Academic Risk Detector - Premium UI
Advanced Streamlit interface with enterprise-grade design
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from gnn_student_risk import HybridGNNModel
from sklearn.preprocessing import RobustScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os

# ============================================================================
# PAGE CONFIGURATION & THEME
# ============================================================================
st.set_page_config(
    page_title="Academic Risk Detector",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    /* Global Styling */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --danger: #ef4444;
        --warning: #f59e0b;
        --success: #10b981;
        --dark: #1f2937;
        --light: #f9fafb;
    }
    
    /* Remove default padding */
    .main {
        padding-top: 2rem;
    }
    
    /* Enhanced Typography */
    h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    h2 {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #6366f1;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.25rem;
        font-weight: 600;
        color: #374151;
    }
    
    h4 {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    
    /* Card Styling */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    .card-success {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05), transparent);
    }
    
    .card-warning {
        border-left: 4px solid #f59e0b;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.05), transparent);
    }
    
    .card-danger {
        border-left: 4px solid #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.05), transparent);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Slider Enhancement */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk Badge */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .risk-badge-safe {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    
    .risk-badge-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
    }
    
    .risk-badge-danger {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
    }
    
    /* Input Enhancement */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background-color: #fafbfc;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        background-color: white;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 1rem 1.5rem;
        background-color: #f3f4f6;
        border: none;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    /* Divider Enhancement */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #d1d5db, transparent);
        margin: 2rem 0;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border-left: 4px solid #6366f1;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Dataframe Enhancement */
    .streamlit-expanderHeader {
        background-color: #f9fafb;
        border-radius: 8px;
    }
    
    /* Scrollable Container */
    .scrollable-container {
        max-height: 500px;
        overflow-y: auto;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        padding: 1rem;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    
    /* Markdown Links */
    a {
        color: #6366f1;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    a:hover {
        color: #8b5cf6;
        text-decoration: underline;
    }
    
    /* Custom Section Headers */
    .section-header {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem;
        }
        
        h2 {
            font-size: 1.5rem;
        }
        
        .card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model.pt'
ORIGINAL_CSV = 'students.csv'
HISTORY_FILE = 'prediction_history.json'

# ============================================================================
# LOAD AND CACHE DATA/MODEL
# ============================================================================
@st.cache_resource
def load_model_and_scaler():
    """Load trained model and fit scaler"""
    df_original = pd.read_csv(ORIGINAL_CSV)
    feature_cols = [col for col in df_original.columns 
                    if col not in ['student_id', 'risk_label']]
    
    scaler = RobustScaler()
    scaler.fit(df_original[feature_cols])
    
    input_dim = len(feature_cols)
    model = HybridGNNModel(
        input_dim=input_dim,
        hidden_dims=[64, 64, 32],
        num_heads=8,
        dropout=0.3
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, scaler, feature_cols

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_student_risk(student_data, model, scaler, feature_cols):
    """Make prediction for a student using graph context"""
    df_original = pd.read_csv(ORIGINAL_CSV)
    X_original = df_original[feature_cols].values
    y_original = df_original['risk_label'].values
    
    X_original_scaled = scaler.transform(X_original)
    
    df_student = pd.DataFrame([student_data])
    X_student_scaled = scaler.transform(df_student[feature_cols])
    
    X_combined = np.vstack([X_original_scaled, X_student_scaled])
    
    from gnn_student_risk import GraphConstructor
    gc = GraphConstructor(features=X_combined, target=np.hstack([y_original, [0]]), k=10)
    
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    edge_index, edge_weights = gc.compute_edge_weights()
    sys.stdout = old_stdout
    
    X_tensor = torch.FloatTensor(X_combined).to(DEVICE)
    edge_index_tensor = torch.LongTensor(edge_index).to(DEVICE)
    
    data = Data(
        x=X_tensor,
        edge_index=edge_index_tensor,
        edge_weight=torch.FloatTensor(edge_weights).to(DEVICE),
        num_nodes=len(X_combined)
    )
    
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    prob = probs[-1]
    return prob

# ============================================================================
# HISTORY MANAGEMENT
# ============================================================================
def load_history():
    """Load prediction history"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_prediction(student_id, prediction, student_data):
    """Save prediction to history"""
    history = load_history()
    history.append({
        'timestamp': datetime.now().isoformat(),
        'student_id': student_id,
        'risk_score': float(prediction),
        'status': 'AT RISK' if prediction > 0.65 else 'SAFE',
        'student_data': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in student_data.items()}
    })
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except:
        pass

# ============================================================================
# HEADER
# ============================================================================
def render_header():
    """Premium header with branding"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 🎓 Academic Risk Detector")
        st.markdown("**✨ AI-powered early warning system for student success** | Powered by Graph Neural Networks")
    with col2:
        status_text = "🟢 Online" if DEVICE.type == 'cuda' else "💻 CPU Mode"
        st.markdown(f"<div class='info-box'>{status_text}</div>", unsafe_allow_html=True)
    
    st.markdown("---")

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    render_header()
    
    model, scaler, feature_cols = load_model_and_scaler()
    
    tab1, tab2, tab3 = st.tabs(["🔮 Predict Risk", "📊 Analytics", "ℹ️ System Info"])
    
    # ========================================================================
    # TAB 1: PREDICT
    # ========================================================================
    with tab1:
        col_header1, col_header2 = st.columns([2, 1])
        with col_header1:
            st.markdown("<div class='section-header'>📝 Student Profile Input</div>", unsafe_allow_html=True)
        with col_header2:
            clear_all = st.button("🔄 Reset Form", use_container_width=True)
        
        # Student ID Section
        st.markdown("##### 👤 Student Information")
        student_id = st.text_input(
            "Student ID",
            placeholder="e.g., STU-2024-001",
            key="student_id_input",
            help="Enter a unique identifier for this student"
        )
        
        # Academic Performance Section
        st.markdown("##### 📚 Academic Performance")
        st.markdown("*Enter grades for each grading period (0-20 scale)*")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Period 1**")
            g1 = st.slider("G1", 0, 20, 10, key="g1_slider",
                          help="First period grade")
            progress_g1 = g1 / 20
            st.progress(progress_g1, text=f"{g1}/20")
        
        with col2:
            st.markdown("**Period 2**")
            g2 = st.slider("G2", 0, 20, 10, key="g2_slider",
                          help="Second period grade")
            progress_g2 = g2 / 20
            st.progress(progress_g2, text=f"{g2}/20")
        
        with col3:
            st.markdown("**Final Grade**")
            g3 = st.slider("G3", 0, 20, 10, key="g3_slider",
                          help="Final grade")
            progress_g3 = g3 / 20
            st.progress(progress_g3, text=f"{g3}/20")
        
        with col4:
            st.markdown("**Average**")
            avg_grade = (g1 + g2 + g3) / 3
            st.markdown(f"""
            <div class='card' style='text-align: center; background: linear-gradient(135deg, #f0f9ff, #e0f2fe);'>
                <div style='font-size: 1.8rem; font-weight: 800; color: #0284c7;'>{avg_grade:.1f}</div>
                <div style='font-size: 0.85rem; color: #6b7280; font-weight: 600;'>Average</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Overall Progress
        overall_progress = avg_grade / 20
        st.progress(overall_progress, text=f"Overall Academic Progress: {overall_progress:.0%}")
        
        st.markdown("---")
        
        # Engagement & Behavior Section
        st.markdown("##### 👥 Engagement & Behavior")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**⏱️ Study Commitment**")
            study_hours = st.slider("Weekly Study Hours", 0.0, 10.0, 2.0, 0.5,
                                   key="study_slider",
                                   help="Average hours spent studying per week")
            study_badge = "🟢 Excellent" if study_hours >= 5 else "🟡 Moderate" if study_hours >= 2 else "🔴 Low"
            st.caption(f"Status: {study_badge}")
        
        with col2:
            st.markdown("**🎯 Engagement Level**")
            engagement_score = st.slider("Engagement Score", 0.0, 3.0, 1.5, 0.1,
                                        key="engagement_slider",
                                        help="Class participation and attentiveness (0-3)")
            engagement_pct = (engagement_score / 3) * 100
            st.progress(engagement_pct / 100, text=f"{engagement_score:.1f}/3.0")
        
        with col3:
            st.markdown("**📍 Attendance**")
            absences = st.slider("Number of Absences", 0, 50, 5,
                                key="absences_slider",
                                help="Total class absences")
            absence_status = "🟢 Good" if absences <= 5 else "🟡 Fair" if absences <= 15 else "🔴 Poor"
            st.caption(f"Status: {absence_status}")
        
        st.markdown("---")
        
        # Additional Metrics
        st.markdown("##### 📊 Academic History & Progress")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Previous Failures**")
            failures = st.slider("Course Failures", 0, 10, 0,
                                key="failures_slider",
                                help="Number of courses previously failed")
            failure_risk = "🟢 None" if failures == 0 else "🟡 Some" if failures <= 2 else "🔴 Multiple"
            st.caption(failure_risk)
        
        with col2:
            st.markdown("**Average Score**")
            avg_score = st.number_input("Overall Average", 0.0, 20.0, 10.0, 0.5,
                                       key="avg_score_input")
            st.progress(avg_score / 20, text=f"{avg_score:.1f}/20")
        
        with col3:
            st.markdown("**Progress Stage**")
            progression = st.selectbox(
                "Course Progression",
                [0, 1, 2],
                format_func=lambda x: {0: "🔴 Early Stage", 1: "🟡 On-Track", 2: "🟢 Advanced"}[x],
                key="progression_select"
            )
        
        with col4:
            st.markdown("**Study Time Category**")
            if study_hours >= 5:
                study_status = "Excellent"
                study_color = "#10b981"
            elif study_hours >= 2:
                study_status = "Good"
                study_color = "#f59e0b"
            else:
                study_status = "Needs Improvement"
                study_color = "#ef4444"
            
            st.markdown(f"""
            <div style='background-color: {study_color}33; border-left: 4px solid {study_color}; padding: 0.75rem; border-radius: 6px;'>
                <div style='font-size: 0.85rem; color: {study_color}; font-weight: 700;'>{study_status.upper()}</div>
                <div style='color: #6b7280;'>{study_hours:.1f}h/week</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Action Buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1.5, 1, 2])
        
        with col_btn1:
            predict_btn = st.button("🔮 PREDICT RISK", use_container_width=True, key="predict_btn")
        
        with col_btn2:
            clear_btn = st.button("🔄 Clear", use_container_width=True)
        
        if clear_btn or clear_all:
            st.rerun()
        
        # ====================================================================
        # RESULTS
        # ====================================================================
        if predict_btn:
            if not student_id:
                st.error("⚠️ Please enter a Student ID before predicting")
            else:
                student_data = {
                    'G1': g1, 'G2': g2, 'G3': g3,
                    'studytime': study_hours, 'absences': absences,
                    'failures': failures, 'progression': float(progression),
                    'avg_score': avg_score, 'engagement_score': engagement_score
                }
                
                with st.spinner("🔄 Analyzing student profile with AI model..."):
                    risk_score = predict_student_risk(student_data, model, scaler, feature_cols)
                
                save_prediction(student_id, risk_score, student_data)
                
                st.markdown("---")
                st.markdown("<div class='section-header'>🎯 Prediction Results</div>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1.2, 1.8, 0.8])
                
                with col1:
                    # Risk Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_score * 100,
                        title={'text': "Risk Score", 'font': {'size': 16}},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        suffix="%",
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#6366f1", 'thickness': 0.7},
                            'steps': [
                                {'range': [0, 33], 'color': "#d1f5ea"},
                                {'range': [33, 66], 'color': "#fed7aa"},
                                {'range': [66, 100], 'color': "#fed7aa"}
                            ],
                            'threshold': {
                                'line': {'color': "#ef4444", 'width': 4},
                                'thickness': 0.75,
                                'value': 65
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=350, margin=dict(l=0, r=0, t=50, b=0))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    # Status & Recommendation
                    if risk_score > 0.65:
                        status = "🔴 HIGH RISK"
                        status_emoji = "🚨"
                        recommendation = """
                        ### ⚡ Immediate Action Required
                        
                        **Priority Actions:**
                        • Schedule urgent meeting with academic advisor
                        • Implement personalized tutoring program
                        • Weekly progress monitoring required
                        • Consider academic support group enrollment
                        • Identify knowledge gaps immediately
                        """
                        color = "danger"
                    elif risk_score > 0.55:
                        status = "🟡 MEDIUM RISK"
                        status_emoji = "⚠️"
                        recommendation = """
                        ### ⚠️ Proactive Support Needed
                        
                        **Recommended Actions:**
                        • Close monitoring of academic progress
                        • Offer optional tutoring support
                        • Suggest peer study groups
                        • Regular check-ins with instructor
                        • Consider study skills workshop
                        """
                        color = "warning"
                    else:
                        status = "🟢 SAFE"
                        status_emoji = "✅"
                        recommendation = """
                        ### ✅ Student On Track
                        
                        **Positive Actions:**
                        • Continue current academic approach
                        • Maintain excellent attendance
                        • Support peer learning initiatives
                        • Explore advanced opportunities
                        • Consider mentoring other students
                        """
                        color = "success"
                    
                    badge_html = f"<span class='risk-badge risk-badge-{color}'>{status}</span>"
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.markdown(f"<div class='card card-{color}'>{recommendation}</div>", 
                               unsafe_allow_html=True)
                
                with col3:
                    st.markdown("**📊 Key Stats**")
                    
                    st.markdown(f"""
                    <div class='card' style='background: linear-gradient(135deg, #f0f9ff, #e0f2fe);'>
                        <div style='text-align: center;'>
                            <div style='font-size: 1.5rem; font-weight: 800; color: #0284c7;'>{risk_score:.0%}</div>
                            <div style='font-size: 0.8rem; color: #6b7280; font-weight: 600;'>RISK SCORE</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("📖 Final Grade", f"{g3}/20")
                    st.metric("📍 Absences", f"{absences} days")
                    st.metric("⏱️ Study Hrs/Wk", f"{study_hours:.1f}h")
                
                # Detailed Profile
                st.markdown("---")
                st.markdown("<div class='section-header'>👤 Detailed Student Profile</div>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📈 Academic Trajectory**")
                    trajectory_data = pd.DataFrame({
                        'Period': ['P1', 'P2', 'Final'],
                        'Grade': [g1, g2, g3]
                    })
                    fig_line = px.line(trajectory_data, x='Period', y='Grade',
                                      title='Grade Progression Over Time',
                                      markers=True, line_shape='spline')
                    fig_line.update_traces(marker_size=10, line_width=3)
                    fig_line.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0),
                                          hovermode='x unified')
                    st.plotly_chart(fig_line, use_container_width=True)
                
                with col2:
                    st.markdown("**📋 Performance Summary**")
                    summary_cols = ['📚 Avg Grade', '⏱️ Study Time', '📍 Absences', 
                                   '🎯 Engagement', '❌ Previous Fails']
                    summary_vals = [
                        f"{avg_grade:.1f}/20",
                        f"{study_hours:.1f}h/wk",
                        f"{absences} days",
                        f"{engagement_score:.1f}/3.0",
                        f"{int(failures)}"
                    ]
                    
                    for label, value in zip(summary_cols, summary_vals):
                        st.metric(label, value)
    
    # ========================================================================
    # TAB 2: ANALYTICS
    # ========================================================================
    with tab2:
        st.markdown("### 📊 Advanced Analytics Dashboard")
        
        history = load_history()
        
        if history:
            # Convert to DataFrame for analysis
            history_df = pd.DataFrame(history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            # ====================================================================
            # KEY METRICS - Top Stats
            # ====================================================================
            st.markdown("#### 📈 Key Performance Indicators")
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            total_predictions = len(history_df)
            at_risk_count = sum([1 for h in history if h['risk_score'] > 0.65])
            medium_count = sum([1 for h in history if 0.55 < h['risk_score'] <= 0.65])
            safe_count = sum([1 for h in history if h['risk_score'] <= 0.55])
            avg_risk = np.mean([h['risk_score'] for h in history])
            max_risk = max([h['risk_score'] for h in history])
            
            with col1:
                st.markdown("""
                <div class='card' style='text-align: center; background: linear-gradient(135deg, #f3f4f6, #e5e7eb);'>
                    <div style='font-size: 2.5rem; font-weight: 800; color: #6366f1;'>{}</div>
                    <div style='font-size: 0.9rem; color: #6b7280; font-weight: 600;'>TOTAL PREDICTIONS</div>
                </div>
                """.format(total_predictions), unsafe_allow_html=True)
            
            with col2:
                delta = "↑" if at_risk_count > 0 else "→"
                st.markdown("""
                <div class='card' style='text-align: center; background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05)); border-left: 4px solid #ef4444;'>
                    <div style='font-size: 2.5rem; font-weight: 800; color: #ef4444;'>{}{}</div>
                    <div style='font-size: 0.9rem; color: #6b7280; font-weight: 600;'>🔴 HIGH RISK</div>
                </div>
                """.format(delta, at_risk_count), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class='card' style='text-align: center; background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05)); border-left: 4px solid #f59e0b;'>
                    <div style='font-size: 2.5rem; font-weight: 800; color: #f59e0b;'>{}</div>
                    <div style='font-size: 0.9rem; color: #6b7280; font-weight: 600;'>🟡 MEDIUM RISK</div>
                </div>
                """.format(medium_count), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class='card' style='text-align: center; background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05)); border-left: 4px solid #10b981;'>
                    <div style='font-size: 2.5rem; font-weight: 800; color: #10b981;'>{}</div>
                    <div style='font-size: 0.9rem; color: #6b7280; font-weight: 600;'>🟢 SAFE</div>
                </div>
                """.format(safe_count), unsafe_allow_html=True)
            
            with col5:
                st.markdown("""
                <div class='card' style='text-align: center; background: linear-gradient(135deg, #fef3c7, #fce8aa);'>
                    <div style='font-size: 2rem; font-weight: 800; color: #92400e;'>{:.0%}</div>
                    <div style='font-size: 0.9rem; color: #6b7280; font-weight: 600;'>AVG RISK</div>
                </div>
                """.format(avg_risk), unsafe_allow_html=True)
            
            with col6:
                st.markdown("""
                <div class='card' style='text-align: center; background: linear-gradient(135deg, #dcfce7, #bbf7d0);'>
                    <div style='font-size: 2rem; font-weight: 800; color: #15803d;'>{:.0%}</div>
                    <div style='font-size: 0.9rem; color: #6b7280; font-weight: 600;'>MAX RISK</div>
                </div>
                """.format(max_risk), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ====================================================================
            # VISUALIZATION SECTION
            # ====================================================================
            st.markdown("#### 📉 Risk Analytics")
            
            col1, col2, col3 = st.columns([2, 1.5, 1.5])
            
            with col1:
                # Time Series - Risk Over Time
                history_df_sorted = history_df.sort_values('timestamp')
                history_df_sorted['index'] = range(len(history_df_sorted))
                
                fig_timeline = go.Figure()
                
                fig_timeline.add_trace(go.Scatter(
                    x=history_df_sorted['index'],
                    y=history_df_sorted['risk_score'],
                    mode='lines+markers',
                    name='Risk Score',
                    line=dict(color='#6366f1', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(99, 102, 241, 0.2)'
                ))
                
                # Add threshold line
                fig_timeline.add_hline(y=0.65, line_dash="dash", line_color="#ef4444", 
                                      annotation_text="High Risk Threshold", annotation_position="right")
                fig_timeline.add_hline(y=0.55, line_dash="dash", line_color="#f59e0b",
                                      annotation_text="Medium Risk Threshold", annotation_position="right")
                
                fig_timeline.update_layout(
                    title='Risk Score Timeline',
                    height=350,
                    xaxis_title='Prediction #',
                    yaxis_title='Risk Score',
                    hovermode='x unified',
                    margin=dict(l=0, r=100, t=40, b=0)
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            with col2:
                # Risk Distribution - Histogram
                risk_scores = [h['risk_score'] for h in history]
                fig_hist = go.Figure(data=[
                    go.Histogram(
                        x=risk_scores,
                        nbinsx=15,
                        marker_color='#8b5cf6',
                        showlegend=False
                    )
                ])
                fig_hist.update_layout(
                    title='Risk Distribution',
                    height=350,
                    xaxis_title='Risk Score',
                    yaxis_title='Count',
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col3:
                # Status Donut
                status_counts = pd.Series([h['status'] for h in history]).value_counts()
                colors_map = {'SAFE': '#10b981', 'AT RISK': '#ef4444'}
                
                fig_donut = go.Figure(data=[go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    hole=.4,
                    marker=dict(colors=[colors_map.get(s, '#6366f1') for s in status_counts.index]),
                    textposition='inside',
                    textinfo='label+percent'
                )])
                fig_donut.update_layout(
                    title='Status Distribution',
                    height=350,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            
            st.markdown("---")
            
            # ====================================================================
            # DETAILED INSIGHTS
            # ====================================================================
            st.markdown("#### 🔍 Detailed Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Risk Statistics**")
                risk_data = {
                    'Metric': ['Minimum', 'Q1 (25%)', 'Median', 'Q3 (75%)', 'Maximum', 'Std Dev'],
                    'Value': [
                        f"{np.min(risk_scores):.1%}",
                        f"{np.percentile(risk_scores, 25):.1%}",
                        f"{np.median(risk_scores):.1%}",
                        f"{np.percentile(risk_scores, 75):.1%}",
                        f"{np.max(risk_scores):.1%}",
                        f"{np.std(risk_scores):.1%}"
                    ]
                }
                st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**🎯 Category Breakdown**")
                category_data = {
                    'Category': ['🔴 High Risk (>65%)', '🟡 Medium Risk (55-65%)', '🟢 Safe (≤55%)'],
                    'Count': [at_risk_count, medium_count, safe_count],
                    'Percentage': [
                        f"{at_risk_count/total_predictions*100:.1f}%",
                        f"{medium_count/total_predictions*100:.1f}%",
                        f"{safe_count/total_predictions*100:.1f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(category_data), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # ====================================================================
            # HEATMAP - Feature Analysis
            # ====================================================================
            st.markdown("#### 🔥 Feature Correlation with Risk")
            
            if 'student_data' in history[0]:
                features_to_analyze = []
                for h in history:
                    if isinstance(h['student_data'], dict):
                        features_to_analyze.append({
                            'G1': h['student_data'].get('G1', 0),
                            'G2': h['student_data'].get('G2', 0),
                            'G3': h['student_data'].get('G3', 0),
                            'study_hours': h['student_data'].get('studytime', 0),
                            'absences': h['student_data'].get('absences', 0),
                            'engagement': h['student_data'].get('engagement_score', 0),
                            'risk_score': h['risk_score']
                        })
                
                if features_to_analyze:
                    feature_df = pd.DataFrame(features_to_analyze)
                    corr_matrix = feature_df.corr()
                    
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate='%{text:.2f}',
                        textfont={"size": 10},
                        colorbar=dict(title="Correlation")
                    ))
                    fig_heatmap.update_layout(
                        title='Feature Correlation Matrix',
                        height=400,
                        width=None
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("---")
            
            # ====================================================================
            # PREDICTION TABLE
            # ====================================================================
            st.markdown("#### 📋 Detailed Prediction History")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_status = st.multiselect(
                    "Filter by Status",
                    ['SAFE', 'AT RISK'],
                    default=['SAFE', 'AT RISK'],
                    key="status_filter"
                )
            
            with col2:
                sort_by = st.selectbox(
                    "Sort by",
                    ['Newest First', 'Oldest First', 'Highest Risk', 'Lowest Risk'],
                    key="sort_select"
                )
            
            with col3:
                display_count = st.number_input("Show rows", 5, len(history), 10, key="display_rows")
            
            # Apply filters
            filtered_df = history_df[history_df['status'].isin(filter_status)].copy()
            
            # Apply sorting
            if sort_by == 'Newest First':
                filtered_df = filtered_df.sort_values('timestamp', ascending=False)
            elif sort_by == 'Oldest First':
                filtered_df = filtered_df.sort_values('timestamp', ascending=True)
            elif sort_by == 'Highest Risk':
                filtered_df = filtered_df.sort_values('risk_score', ascending=False)
            elif sort_by == 'Lowest Risk':
                filtered_df = filtered_df.sort_values('risk_score', ascending=True)
            
            filtered_df = filtered_df.head(display_count).copy()
            filtered_df['timestamp'] = filtered_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            filtered_df['risk_score'] = filtered_df['risk_score'].apply(lambda x: f"{x:.1%}")
            
            display_cols = ['timestamp', 'student_id', 'risk_score', 'status']
            
            st.dataframe(
                filtered_df[display_cols].rename(columns={
                    'timestamp': '📅 Time',
                    'student_id': '👤 Student ID',
                    'risk_score': '📊 Risk Score',
                    'status': '🎯 Status'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            
            # ====================================================================
            # EXPORT OPTIONS
            # ====================================================================
            st.markdown("#### 💾 Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            # CSV Export
            with col1:
                csv_data = history_df.copy()
                csv_data['timestamp'] = csv_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                csv = csv_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # JSON Export
            with col2:
                json_data = json.dumps(history, indent=2, default=str)
                st.download_button(
                    label="📤 Download JSON",
                    data=json_data,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Summary Report
            with col3:
                summary_text = f"""
PREDICTION ANALYTICS SUMMARY
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
Total Predictions: {total_predictions}
Date Range: {history_df['timestamp'].min().strftime('%Y-%m-%d')} to {history_df['timestamp'].max().strftime('%Y-%m-%d')}

RISK DISTRIBUTION
🔴 High Risk (>65%): {at_risk_count} ({at_risk_count/total_predictions*100:.1f}%)
🟡 Medium Risk (55-65%): {medium_count} ({medium_count/total_predictions*100:.1f}%)
🟢 Safe (≤55%): {safe_count} ({safe_count/total_predictions*100:.1f}%)

STATISTICS
Average Risk: {avg_risk:.1%}
Median Risk: {np.median(risk_scores):.1%}
Max Risk: {max_risk:.1%}
Min Risk: {np.min(risk_scores):.1%}
Std Dev: {np.std(risk_scores):.1%}
                """
                st.download_button(
                    label="📄 Download Report",
                    data=summary_text,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        else:
            st.info("📭 No predictions yet. Make your first prediction in the **Predict Risk** tab!")
    
    # ========================================================================
    # TAB 3: SYSTEM INFO
    # ========================================================================
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 Model Information")
            st.markdown("""
            **Architecture:** Hybrid GNN (GCN + GAT)
            - GCN Layer 1: 9 → 64 channels
            - GAT Layer: 64 → 64 channels (8 heads)
            - GCN Layer 2: 64 → 32 channels
            - Output: 32 → 1 (binary classification)
            
            **Parameters:** 36,929
            **Optimization:** Adam + ReduceLROnPlateau
            **Loss Function:** Weighted BCE + Focal Loss
            """)
        
        with col2:
            st.markdown("### 📊 Performance Metrics")
            metrics_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Train': ['94.3%', '90.4%', '99.0%', '94.6%', '97.9%'],
                'Test': ['93.7%', '92.7%', '98.1%', '95.3%', '99.0%']
            }
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ System Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Device", "GPU" if DEVICE.type == 'cuda' else "CPU")
        
        with col2:
            st.metric("Model Status", "✅ Ready")
        
        with col3:
            st.metric("Data Points", "395 students")
        
        st.markdown("---")
        
        st.markdown("### 📚 Feature Guide")
        st.markdown("""
        | Feature | Range | Description |
        |---------|-------|-------------|
        | **G1** | 0-20 | First period grade |
        | **G2** | 0-20 | Second period grade |
        | **G3** | 0-20 | Final grade |
        | **Study Hours** | 0-10 | Weekly study commitment |
        | **Absences** | 0-50 | Total class absences |
        | **Engagement** | 0-3 | Class participation level |
        | **Failures** | 0-10 | Previous course failures |
        | **Progression** | 0-2 | Course progress stage |
        """)

if __name__ == "__main__":
    main()
