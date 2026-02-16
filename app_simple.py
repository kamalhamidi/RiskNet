"""
Student Academic Risk Detector - Simple Version
A clean, easy-to-understand Streamlit app for predicting student dropout risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from gnn_student_risk import HybridGNNModel, GraphConstructor
from sklearn.preprocessing import RobustScaler
import plotly.graph_objects as go
import json
import os
from datetime import datetime

# Page setup
st.set_page_config(page_title="Student Risk Detector", page_icon="🎓", layout="wide")

# Constants
DEVICE = torch.device('cpu')
MODEL_PATH = 'best_model.pt'
DATA_CSV = 'students.csv'
HISTORY_FILE = 'prediction_history.json'

# Load model and data
@st.cache_resource
def load_model():
    """Load the trained GNN model"""
    df = pd.read_csv(DATA_CSV)
    feature_cols = [col for col in df.columns if col not in ['student_id', 'risk_label']]
    
    scaler = RobustScaler()
    scaler.fit(df[feature_cols])
    
    model = HybridGNNModel(
        input_dim=len(feature_cols),
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

# Prediction function
def predict_risk(student_data, model, scaler, feature_cols):
    """Make a prediction for a student"""
    # Load original data
    df_original = pd.read_csv(DATA_CSV)
    X_original = df_original[feature_cols].values
    y_original = df_original['risk_label'].values
    
    # Scale data
    X_original_scaled = scaler.transform(X_original)
    df_student = pd.DataFrame([student_data])
    X_student_scaled = scaler.transform(df_student[feature_cols])
    
    # Combine data
    X_combined = np.vstack([X_original_scaled, X_student_scaled])
    
    # Build graph
    gc = GraphConstructor(features=X_combined, target=np.hstack([y_original, [0]]), k=10)
    
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    edge_index, edge_weights = gc.compute_edge_weights()
    sys.stdout = old_stdout
    
    # Make prediction
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
    
    return probs[-1]

# History management
def load_history():
    """Load prediction history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(student_id, risk_score, student_data):
    """Save prediction to history"""
    history = load_history()
    history.append({
        'timestamp': datetime.now().isoformat(),
        'student_id': student_id,
        'risk_score': float(risk_score),
        'status': 'AT RISK' if risk_score > 0.65 else 'SAFE',
        'student_data': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in student_data.items()}
    })
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except:
        pass

# ============================================================================
# MAIN APP
# ============================================================================

st.title("🎓 Student Academic Risk Detector")
st.write("A simple tool to predict if a student is at risk of dropping out")

# Load model
model, scaler, feature_cols = load_model()

# Tabs
tab1, tab2 = st.tabs(["📊 Make Prediction", "📈 History"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    st.subheader("Enter Student Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Student ID**")
        student_id = st.text_input("Student ID", placeholder="e.g., STU-001")
    
    with col2:
        st.write("**Grades (0-20)**")
        col_g = st.columns(3)
        with col_g[0]:
            g1 = st.number_input("G1", min_value=0, max_value=20, value=10)
        with col_g[1]:
            g2 = st.number_input("G2", min_value=0, max_value=20, value=10)
        with col_g[2]:
            g3 = st.number_input("G3", min_value=0, max_value=20, value=10)
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Engagement & Attendance**")
        study_hours = st.slider("Study hours per week", 0.0, 10.0, 2.0)
        absences = st.slider("Number of absences", 0, 50, 5)
        engagement = st.slider("Engagement score", 0.0, 3.0, 1.5)
    
    with col2:
        st.write("**Academic History**")
        failures = st.slider("Previous failures", 0, 10, 0)
        avg_score = st.slider("Average score", 0.0, 20.0, 10.0)
        progression = st.selectbox("Course progression", ["Early", "On-Track", "Advanced"], index=1)
    
    st.write("---")
    
    # Predict button
    if st.button("🔮 Predict Risk", use_container_width=True):
        if not student_id:
            st.error("Please enter a Student ID")
        else:
            # Prepare data
            student_data = {
                'G1': g1, 'G2': g2, 'G3': g3,
                'studytime': study_hours, 'absences': absences,
                'failures': failures, 'progression': float(['Early', 'On-Track', 'Advanced'].index(progression)),
                'avg_score': avg_score, 'engagement_score': engagement
            }
            
            # Make prediction
            with st.spinner("Analyzing..."):
                risk_score = predict_risk(student_data, model, scaler, feature_cols)
            
            # Save to history
            save_history(student_id, risk_score, student_data)
            
            # Show results
            st.write("---")
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    title="Risk Score",
                    suffix="%",
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgreen"},
                            {'range': [33, 66], 'color': "lightyellow"},
                            {'range': [66, 100], 'color': "lightcoral"}
                        ]
                    }
                ))
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Status and recommendations
                if risk_score > 0.65:
                    st.error(f"🔴 **HIGH RISK** ({risk_score:.0%})")
                    st.write("""
                    **Recommendations:**
                    - Meet with academic advisor
                    - Get tutoring support
                    - Weekly check-ins
                    - Study groups
                    """)
                elif risk_score > 0.55:
                    st.warning(f"🟡 **MEDIUM RISK** ({risk_score:.0%})")
                    st.write("""
                    **Recommendations:**
                    - Monitor progress
                    - Optional tutoring
                    - Study groups
                    - Regular check-ins
                    """)
                else:
                    st.success(f"🟢 **SAFE** ({risk_score:.0%})")
                    st.write("""
                    **Recommendations:**
                    - Continue current approach
                    - Maintain attendance
                    - Help other students
                    - Keep up the work
                    """)
            
            st.write("---")
            
            # Details
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Grades Chart**")
                df_grades = pd.DataFrame({
                    'Period': ['P1', 'P2', 'Final'],
                    'Grade': [g1, g2, g3]
                })
                fig_line = go.Figure(data=go.Scatter(x=df_grades['Period'], y=df_grades['Grade'], 
                                                     mode='lines+markers'))
                fig_line.update_layout(height=250)
                st.plotly_chart(fig_line, use_container_width=True)
            
            with col2:
                st.write("**Student Summary**")
                st.metric("Average Grade", f"{(g1+g2+g3)/3:.1f}/20")
                st.metric("Study Hours/Week", f"{study_hours:.1f}")
                st.metric("Absences", f"{int(absences)}")
                st.metric("Engagement", f"{engagement:.1f}/3.0")

# ============================================================================
# TAB 2: HISTORY
# ============================================================================
with tab2:
    st.subheader("Prediction History")
    
    history = load_history()
    
    if history:
        # Statistics
        total = len(history)
        at_risk = sum(1 for h in history if h['risk_score'] > 0.65)
        safe = sum(1 for h in history if h['risk_score'] <= 0.65)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", total)
        col2.metric("🔴 At Risk", at_risk)
        col3.metric("🟢 Safe", safe)
        
        st.write("---")
        
        # Chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution
            risk_scores = [h['risk_score'] for h in history]
            fig_hist = go.Figure(data=[go.Histogram(x=risk_scores, nbinsx=15)])
            fig_hist.update_layout(title="Risk Score Distribution", height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Status pie chart
            statuses = pd.Series([h['status'] for h in history]).value_counts()
            fig_pie = go.Figure(data=[go.Pie(labels=statuses.index, values=statuses.values)])
            fig_pie.update_layout(title="Status Distribution", height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.write("---")
        
        # Table
        st.write("**All Predictions**")
        df_history = pd.DataFrame(history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        df_history['risk_score'] = df_history['risk_score'].apply(lambda x: f"{x:.0%}")
        
        st.dataframe(
            df_history[['timestamp', 'student_id', 'risk_score', 'status']].rename(
                columns={'timestamp': 'Time', 'student_id': 'Student', 'risk_score': 'Risk', 'status': 'Status'}
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Download
        csv = df_history.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, 
                          f"predictions_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    else:
        st.info("No predictions yet. Make one in the first tab!")

