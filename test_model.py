"""
Test Script: GNN Student Risk Prediction Model (SIMPLIFIED)
Tests the trained model with synthetic graph data on new students
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from gnn_student_risk import (
    DataAnalyzer, GraphConstructor, HybridGNNModel, 
    RobustEvaluator
)
from sklearn.preprocessing import RobustScaler

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model.pt'
ORIGINAL_CSV = 'students.csv'

print("=" * 70)
print("🧪 TEST: GNN Student Risk Prediction Model")
print("=" * 70)
print(f"Device: {DEVICE}\n")

# ============================================================================
# STEP 1: LOAD ORIGINAL DATA FOR SCALING PARAMETERS
# ============================================================================
print("[1] Loading original dataset for preprocessing parameters...")
df_original = pd.read_csv(ORIGINAL_CSV)

# Get the correct feature columns
feature_cols = [col for col in df_original.columns 
                if col not in ['student_id', 'risk_label']]

print(f"   ✓ Loaded {len(df_original)} original students")
print(f"   ✓ Features ({len(feature_cols)}): {feature_cols}\n")

# Fit scaler on original data
scaler = RobustScaler()
scaler.fit(df_original[feature_cols])
print(f"   ✓ Scaler fitted on original data\n")

# ============================================================================
# STEP 2: CREATE TEST STUDENTS
# ============================================================================
print("[2] Creating 5 fictional student profiles for testing...")

# Create realistic student data with the correct columns
test_students = pd.DataFrame({
    'G1': [6, 15, 8, 14, 5],
    'G2': [5, 14, 7, 13, 4],
    'G3': [6, 15, 8, 14, 5],
    'studytime': [1, 3, 2, 3, 1],
    'absences': [8, 2, 5, 1, 15],
    'failures': [1, 0, 0, 0, 3],
    'progression': [0, 1, 1, 1, 0],
    'avg_score': [5.7, 14.7, 7.7, 13.7, 4.7],
    'engagement_score': [1.2, 2.8, 1.8, 2.6, 0.9],
    'student_id': [401, 402, 403, 404, 405]
})

print("   Student Profiles:")
print("   " + "─" * 70)
for idx, row in test_students.iterrows():
    student_id = int(row['student_id'])
    final_grade = row['G3']
    absences = int(row['absences'])
    status_pred = "🔴 LIKELY AT RISK" if final_grade < 8 else "🟢 LIKELY SAFE"
    print(f"   Student {student_id}: Grade={final_grade:.0f}, Absences={absences} {status_pred}")
print("   " + "─" * 70 + "\n")

# ============================================================================
# STEP 3: SCALE TEST DATA
# ============================================================================
print("[3] Scaling test student data...")
X_test_scaled = scaler.transform(test_students[feature_cols])
X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
print(f"   ✓ Scaled shape: {X_test_tensor.shape}")
print(f"   ✓ Device: {X_test_tensor.device}\n")

# ============================================================================
# STEP 4: CREATE SYNTHETIC GRAPH FOR TEST STUDENTS
# ============================================================================
print("[4] Creating synthetic graph structure for test students...")
n_test = len(test_students)

# Create simple connectivity (each student connected to similar ones)
# For simplicity: create a fully connected graph on test set
edges = []
for i in range(n_test):
    for j in range(n_test):
        if i != j:
            edges.append([i, j])

if edges:
    edge_index = torch.LongTensor(edges).t().contiguous().to(DEVICE)
else:
    # Fallback: self-loops only
    edge_index = torch.LongTensor([[i, i] for i in range(n_test)]).t().contiguous().to(DEVICE)

print(f"   ✓ Graph created: {n_test} nodes, {edge_index.shape[1]} edges")
print(f"   ✓ Edge index shape: {edge_index.shape}\n")

# ============================================================================
# STEP 5: LOAD TRAINED MODEL
# ============================================================================
print(f"[5] Loading trained model from '{MODEL_PATH}'...")
try:
    # Initialize model architecture
    input_dim = len(feature_cols)
    model = HybridGNNModel(
        input_dim=input_dim,
        hidden_dims=[64, 64, 32],
        num_heads=8,
        dropout=0.3
    ).to(DEVICE)
    
    # Load state dict
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Model device: {next(model.parameters()).device}\n")
except Exception as e:
    print(f"   ✗ ERROR loading model: {e}\n")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# STEP 6: CREATE PYTORCH GEOMETRIC DATA OBJECT
# ============================================================================
print("[6] Creating PyTorch Geometric data object...")
test_data = Data(
    x=X_test_tensor,
    edge_index=edge_index,
    num_nodes=n_test
)
print(f"   ✓ Data object created")
print(f"   ✓ x shape: {test_data.x.shape}")
print(f"   ✓ edge_index shape: {test_data.edge_index.shape}\n")

# ============================================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================================
print("[7] Making predictions on test students...")
print("   " + "─" * 70)

predictions_proba = []
with torch.no_grad():
    logit = model(test_data)
    probs = torch.sigmoid(logit).squeeze().cpu().numpy()
    if probs.ndim == 0:
        predictions_proba = [probs.item()]
    else:
        predictions_proba = probs.tolist()

predictions_class = [1 if p > 0.5 else 0 for p in predictions_proba]

# ============================================================================
# STEP 8: DISPLAY RESULTS
# ============================================================================
print("\n   PREDICTION RESULTS:")
print("   " + "─" * 70)
print(f"   {'Student':<10} {'Grade':<8} {'Risk%':<10} {'Decision':<12} {'Status':<20}")
print("   " + "─" * 70)

for idx, row in test_students.iterrows():
    student_id = int(row['student_id'])
    grade = row['G3']
    prob = predictions_proba[idx]
    pred_class = "AT RISK" if predictions_class[idx] == 1 else "SAFE"
    
    # Color coding
    if predictions_class[idx] == 1:
        status_emoji = "🔴"
        status_text = f"AT RISK ({prob:.1%})"
    else:
        status_emoji = "🟢"
        status_text = f"SAFE ({1-prob:.1%})"
    
    print(f"   {student_id:<10} {grade:<8.0f} {prob:<10.1%} {pred_class:<12} {status_emoji} {status_text}")

print("   " + "─" * 70)

# ============================================================================
# STEP 9: SUMMARY STATISTICS
# ============================================================================
print("\n[8] Summary Statistics:")
print("   " + "─" * 70)
at_risk_count = sum(predictions_class)
safe_count = len(predictions_class) - at_risk_count
avg_risk_score = np.mean(predictions_proba)
std_risk_score = np.std(predictions_proba)

print(f"   Total predictions: {len(predictions_class)}")
print(f"   Students at risk:  {at_risk_count} ({at_risk_count/len(predictions_class):.0%})")
print(f"   Students safe:     {safe_count} ({safe_count/len(predictions_class):.0%})")
print(f"   Average risk score: {avg_risk_score:.1%}")
print(f"   Std dev:            {std_risk_score:.1%}")
print("   " + "─" * 70)

# ============================================================================
# STEP 10: EXPORT RESULTS
# ============================================================================
print("\n[9] Exporting results...")
results_df = test_students[['student_id', 'G3', 'absences']].copy()
results_df['predicted_risk_score'] = predictions_proba
results_df['predicted_class'] = ['AT RISK' if c == 1 else 'SAFE' for c in predictions_class]
results_df['confidence'] = [max(p, 1-p) for p in predictions_proba]

output_path = 'test_predictions.csv'
results_df.to_csv(output_path, index=False)
print(f"   ✓ Results saved to '{output_path}'")
print(f"   ✓ Shape: {results_df.shape}\n")

print("=" * 70)
print("✅ TEST COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\n📊 Generated file:")
print(f"   - test_predictions.csv (detailed predictions)")
print("\n📋 Interpretation:")
print("   - Risk Score > 0.5 → Student AT RISK (needs intervention)")
print("   - Risk Score ≤ 0.5 → Student SAFE (on track)")
print("\n")
