"""
Streamlit Web Interface for GNN Student Risk Prediction
Interactive dashboard for predicting student academic risk
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
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🎓 Student Risk Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    # Load original data for scaler
    df_original = pd.read_csv(ORIGINAL_CSV)
    feature_cols = [col for col in df_original.columns 
                    if col not in ['student_id', 'risk_label']]
    
    # Fit scaler
    scaler = RobustScaler()
    scaler.fit(df_original[feature_cols])
    
    # Initialize and load model
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

@st.cache_data
def get_feature_info():
    """Get feature descriptions and ranges"""
    return {
        'G1': {'name': 'First Period Grade', 'min': 0, 'max': 20, 'step': 1},
        'G2': {'name': 'Second Period Grade', 'min': 0, 'max': 20, 'step': 1},
        'G3': {'name': 'Final Grade', 'min': 0, 'max': 20, 'step': 1},
        'studytime': {'name': 'Weekly Study Hours', 'min': 0, 'max': 10, 'step': 0.5},
        'absences': {'name': 'Number of Absences', 'min': 0, 'max': 50, 'step': 1},
        'failures': {'name': 'Previous Failures', 'min': 0, 'max': 10, 'step': 1},
        'progression': {'name': 'Course Progression', 'min': 0, 'max': 2, 'step': 1},
        'avg_score': {'name': 'Average Score', 'min': 0, 'max': 20, 'step': 0.1},
        'engagement_score': {'name': 'Engagement Score', 'min': 0, 'max': 3, 'step': 0.1},
    }

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_student_risk(student_data, model, scaler, feature_cols):
    """Make prediction for a student using graph context"""
    # Load original training data for graph context
    df_original = pd.read_csv(ORIGINAL_CSV)
    X_original = df_original[feature_cols].values
    y_original = df_original['risk_label'].values
    
    # Scale original data
    X_original_scaled = scaler.transform(X_original)
    
    # Convert new student to DataFrame and scale
    df_student = pd.DataFrame([student_data])
    X_student_scaled = scaler.transform(df_student[feature_cols])
    
    # Combine: new student + original students
    X_combined = np.vstack([X_original_scaled, X_student_scaled])
    
    # Create graph on combined data
    from gnn_student_risk import GraphConstructor
    gc = GraphConstructor(features=X_combined, target=np.hstack([y_original, [0]]), k=10)
    
    # Suppress print output
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    edge_index, edge_weights = gc.compute_edge_weights()
    sys.stdout = old_stdout
    
    # Create tensor
    X_tensor = torch.FloatTensor(X_combined).to(DEVICE)
    edge_index_tensor = torch.LongTensor(edge_index).to(DEVICE)
    
    # Create data object
    data = Data(
        x=X_tensor,
        edge_index=edge_index_tensor,
        edge_weight=torch.FloatTensor(edge_weights).to(DEVICE),
        num_nodes=len(X_combined)
    )
    
    # Get predictions
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # Return prediction for the new student (last one)
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
    # Convert numpy types to Python native types for JSON serialization
    history.append({
        'timestamp': datetime.now().isoformat(),
        'student_id': student_id,
        'risk_score': float(prediction),  # Convert to float
        'status': 'AT RISK' if prediction > 0.65 else 'SAFE',
        'student_data': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in student_data.items()}
    })
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except:
        pass  # Silently fail if history can't be saved

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Load model and data
    model, scaler, feature_cols = load_model_and_scaler()
    feature_info = get_feature_info()
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🎓 Student Academic Risk Detector</h1>
        <p style='font-size: 18px; color: #666;'>
            Powered by Graph Neural Networks | Predict student dropout risk
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 History", "ℹ️ About"])
    
    # ========================================================================
    # TAB 1: PREDICT
    # ========================================================================
    with tab1:
        st.markdown("### Enter Student Information")
        
        col1, col2, col3 = st.columns(3)
        
        # Student ID
        with col1:
            student_id = st.text_input(
                "Student ID",
                value="",
                placeholder="e.g., S001"
            )
        
        # Layout for input fields
        col1, col2 = st.columns(2)
        
        student_data = {}
        
        with col1:
            st.markdown("#### Academic Performance")
            student_data['G1'] = st.slider(
                "First Period Grade",
                min_value=0,
                max_value=20,
                value=10,
                step=1,
                help="Grade in first grading period (0-20)"
            )
            student_data['G2'] = st.slider(
                "Second Period Grade",
                min_value=0,
                max_value=20,
                value=10,
                step=1,
                help="Grade in second grading period (0-20)"
            )
            student_data['G3'] = st.slider(
                "Final Grade",
                min_value=0,
                max_value=20,
                value=10,
                step=1,
                help="Final grade (0-20)"
            )
            student_data['avg_score'] = st.slider(
                "Average Score",
                min_value=0.0,
                max_value=20.0,
                value=10.0,
                step=0.1
            )
            student_data['failures'] = st.slider(
                "Previous Failures",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Number of previous course failures"
            )
        
        with col2:
            st.markdown("#### Engagement & Behavior")
            student_data['studytime'] = st.slider(
                "Weekly Study Hours",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help="Hours spent studying per week"
            )
            student_data['absences'] = st.slider(
                "Number of Absences",
                min_value=0,
                max_value=50,
                value=5,
                step=1,
                help="Number of class absences"
            )
            student_data['engagement_score'] = st.slider(
                "Engagement Score",
                min_value=0.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Class participation and engagement (0-3)"
            )
            student_data['progression'] = st.selectbox(
                "Course Progression",
                options=[0, 1, 2],
                format_func=lambda x: {0: "Early", 1: "On-Track", 2: "Late"}[x],
                help="Progress through the course"
            )
            student_data['progression'] = float(student_data['progression'])
        
        # Prediction Button
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            predict_btn = st.button("🔮 PREDICT RISK", use_container_width=True)
        
        with col_btn2:
            clear_btn = st.button("🔄 Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        # ====================================================================
        # RESULTS
        # ====================================================================
        if predict_btn:
            if not student_id:
                st.warning("⚠️ Please enter a Student ID")
            else:
                # Make prediction
                with st.spinner("🔄 Analyzing student profile..."):
                    risk_score = predict_student_risk(student_data, model, scaler, feature_cols)
                
                # Save to history
                save_prediction(student_id, risk_score, student_data)
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    # Risk meter
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=risk_score * 100,
                        title={'text': "Risk Score (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 65  # Updated threshold
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=350, margin=dict(l=0, r=0, t=50, b=0))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    # Status and recommendation (Updated threshold: 0.65)
                    if risk_score > 0.65:
                        status = "🔴 HIGH RISK"
                        recommendation = "Immediate intervention needed. Schedule meeting with academic advisor."
                        color = "#ff0000"
                    elif risk_score > 0.55:
                        status = "🟡 MEDIUM RISK"
                        recommendation = "Monitor closely. Suggest tutoring or academic support."
                        color = "#ffaa00"
                    else:
                        status = "🟢 SAFE"
                        recommendation = "Student is on track. Continue current path."
                        color = "#00aa00"
                    
                    st.markdown(f"""
                    <div style='
                        background-color: {color}20;
                        border-left: 5px solid {color};
                        padding: 20px;
                        border-radius: 5px;
                    '>
                        <h3 style='color: {color}; margin: 0;'>{status}</h3>
                        <p style='margin: 10px 0 0 0; font-size: 14px;'>{recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Key metrics
                    st.markdown("#### Key Metrics")
                    metric_cols = st.columns(2)
                    with metric_cols[0]:
                        st.metric("Risk Score", f"{risk_score:.1%}")
                        st.metric("Grade Avg", f"{student_data['G3']:.1f}/20")
                    with metric_cols[1]:
                        st.metric("Absences", int(student_data['absences']))
                        st.metric("Study Hours", f"{student_data['studytime']:.1f}h/week")
                
                # Feature importance visualization
                st.markdown("---")
                st.markdown("#### 📈 Student Profile Summary")
                
                summary_data = {
                    'Feature': ['Grades (G1/G2/G3)', 'Study Time', 'Absences', 'Engagement', 'Failures'],
                    'Value': [
                        f"{student_data['G1']}/{student_data['G2']}/{student_data['G3']}",
                        f"{student_data['studytime']:.1f} hrs",
                        f"{int(student_data['absences'])}",
                        f"{student_data['engagement_score']:.1f}/3",
                        f"{int(student_data['failures'])}"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # TAB 2: HISTORY
    # ========================================================================
    with tab2:
        st.markdown("### 📋 Prediction History")
        
        history = load_history()
        
        if history:
            # Convert to DataFrame
            history_df = pd.DataFrame(history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            history_df['risk_score'] = history_df['risk_score'].apply(lambda x: f"{x:.1%}")
            
            # Display stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", len(history))
            with col2:
                at_risk = sum([1 for h in history if h['risk_score'] > 0.5])
                st.metric("At Risk", at_risk)
            with col3:
                safe = len(history) - at_risk
                st.metric("Safe", safe)
            with col4:
                avg_risk = np.mean([h['risk_score'] for h in history])
                st.metric("Avg Risk", f"{avg_risk:.1%}")
            
            # Display table
            st.markdown("---")
            display_cols = ['timestamp', 'student_id', 'risk_score', 'status']
            st.dataframe(
                history_df[display_cols].rename(columns={
                    'timestamp': '📅 Time',
                    'student_id': '👤 Student ID',
                    'risk_score': '📊 Risk Score',
                    'status': '🎯 Status'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History as CSV",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("📭 No predictions yet. Go to 'Predict' tab to make your first prediction!")
    
    # ========================================================================
    # TAB 3: ABOUT
    # ========================================================================
    with tab3:
        st.markdown("""
        ### 🎓 About This System
        
        This is an AI-powered system for predicting student academic risk using **Graph Neural Networks (GNN)**.
        
        #### 🔬 Technology Stack
        - **Architecture**: Hybrid GNN (GCN + GAT)
        - **Framework**: PyTorch + PyTorch Geometric
        - **Performance**: 99% ROC-AUC, 100% Recall
        
        #### 📊 Features Analyzed
        - **Academic**: Grades (G1, G2, G3), Average Score
        - **Engagement**: Study Time, Participation, Absences
        - **Progress**: Course Progression, Previous Failures
        
        #### 🎯 Risk Categories
        - **🔴 High Risk (>70%)**: Immediate intervention needed
        - **🟡 Medium Risk (50-70%)**: Monitoring & support recommended
        - **🟢 Safe (<50%)**: Student on track
        
        #### 📈 How It Works
        1. Student features are preprocessed and scaled
        2. Graph relationships between students are constructed
        3. GNN analyzes patterns across the entire student network
        4. Risk prediction is generated with confidence score
        
        #### 📚 Model Validation
        - **Training Set**: 316 students
        - **Test Set**: 79 students
        - **Accuracy**: 93.67%
        - **Precision**: 91.23%
        - **Recall**: 100.00%
        - **F1-Score**: 95.41%
        
        #### 💡 Recommendations
        - Use this tool as support, not replacement for human judgment
        - Combine with other assessment methods
        - Follow up with identified students
        - Regularly validate predictions against outcomes
        
        ---
        *Developed with Advanced AI Research | 2026*
        """)

if __name__ == "__main__":
    main()
