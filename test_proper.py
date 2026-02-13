"""
Proper Test Script: GNN Student Risk Prediction Model
Tests the model using the actual training graph structure
"""

import torch
import numpy as np
import pandas as pd
from gnn_student_risk import (
    DataAnalyzer, GraphConstructor, HybridGNNModel, 
    RobustEvaluator
)
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model.pt'
ORIGINAL_CSV = 'students.csv'
SEED = 42

print("=" * 80)
print("🧪 PROPER TEST: GNN Model Validation on Original Graph")
print("=" * 80)
print(f"Device: {DEVICE}\n")

# ============================================================================
# STEP 1: LOAD AND ANALYZE DATA
# ============================================================================
print("[1] Loading and analyzing data...")
df = pd.read_csv(ORIGINAL_CSV)
print(f"   ✓ Loaded {len(df)} students")

# Get feature columns
feature_cols = [col for col in df.columns 
                if col not in ['student_id', 'risk_label']]
print(f"   ✓ Features: {feature_cols}")

# Extract X and y
X = df[feature_cols].values
y = df['risk_label'].values
print(f"   ✓ Class distribution: {np.sum(y)} at-risk, {len(y) - np.sum(y)} safe\n")

# ============================================================================
# STEP 2: SCALE DATA
# ============================================================================
print("[2] Scaling features...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
print(f"   ✓ Features scaled with RobustScaler\n")

# ============================================================================
# STEP 3: CREATE GRAPH
# ============================================================================
print("[3] Creating student similarity graph...")
graph_constructor = GraphConstructor(features=X_scaled, target=y, k=10)
edge_index, edge_weights = graph_constructor.compute_edge_weights()
print(f"   ✓ Graph created: {len(df)} nodes, {edge_index.shape[1]} edges")
print(f"   ✓ Edge density: {edge_index.shape[1] / (len(df) * len(df)):.2%}\n")

# ============================================================================
# STEP 4: TRAIN/TEST SPLIT (STRATIFIED)
# ============================================================================
print("[4] Creating train/test split (80/20 stratified)...")
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, np.arange(len(y)), 
    test_size=0.2, 
    random_state=SEED, 
    stratify=y
)
print(f"   ✓ Train set: {len(X_train)} students ({np.sum(y_train)} at-risk)")
print(f"   ✓ Test set: {len(X_test)} students ({np.sum(y_test)} at-risk)\n")

# ============================================================================
# STEP 5: LOAD MODEL
# ============================================================================
print("[5] Loading trained model...")
try:
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
    print(f"   ✓ Model loaded successfully\n")
except Exception as e:
    print(f"   ✗ ERROR: {e}\n")
    exit(1)

# ============================================================================
# STEP 6: PREPARE DATA FOR INFERENCE
# ============================================================================
print("[6] Preparing graph data for inference...")
import torch
from torch_geometric.data import Data

X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
edge_index_tensor = torch.LongTensor(edge_index).to(DEVICE)
edge_weights_tensor = torch.FloatTensor(edge_weights).to(DEVICE)

# Create test mask
test_mask = torch.zeros(len(df), dtype=torch.bool)
test_mask[idx_test] = True

pyg_data = Data(
    x=X_tensor,
    edge_index=edge_index_tensor,
    edge_weight=edge_weights_tensor,
    num_nodes=len(df)
)
print(f"   ✓ PyG data created")
print(f"   ✓ Shape: {pyg_data.x.shape}\n")

# ============================================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================================
print("[7] Making predictions on full graph...")
with torch.no_grad():
    logits = model(pyg_data)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

# Filter to test set
y_pred_proba_test = probs[idx_test]
y_pred_test = (y_pred_proba_test > 0.5).astype(int)

print(f"   ✓ Predictions made")
print(f"   ✓ Test set predictions: {np.sum(y_pred_test)} at-risk, {len(y_pred_test) - np.sum(y_pred_test)} safe\n")

# ============================================================================
# STEP 8: EVALUATE METRICS
# ============================================================================
print("[8] Computing evaluation metrics...")
print("   " + "─" * 75)

# Calculate metrics
acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, zero_division=0)
rec = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba_test)

print(f"   Accuracy:   {acc:.2%}")
print(f"   Precision:  {prec:.2%}")
print(f"   Recall:     {rec:.2%}")
print(f"   F1-Score:   {f1:.2%}")
print(f"   ROC-AUC:    {auc:.2%}")
print("   " + "─" * 75 + "\n")

# ============================================================================
# STEP 9: DETAILED ANALYSIS
# ============================================================================
print("[9] Detailed Classification Report...")
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred_test)
print(classification_report(y_test, y_pred_test, 
                           target_names=['Safe', 'At Risk']))

tn, fp, fn, tp = cm.ravel()
print(f"   Confusion Matrix:")
print(f"   ├─ True Negatives:  {tn}")
print(f"   ├─ False Positives: {fp}")
print(f"   ├─ False Negatives: {fn}")
print(f"   └─ True Positives:  {tp}\n")

# ============================================================================
# STEP 10: EXAMPLE PREDICTIONS
# ============================================================================
print("[10] Example Predictions on Test Set (showing 15 random samples)...")
print("   " + "─" * 75)
print(f"   {'Student ID':<12} {'Actual':<8} {'Predicted':<12} {'Risk Score':<12} {'Confidence':<12}")
print("   " + "─" * 75)

# Get random test indices
np.random.seed(SEED)
sample_indices = np.random.choice(len(idx_test), min(15, len(idx_test)), replace=False)

for i in sample_indices:
    student_id = df.iloc[idx_test[i]]['student_id']
    actual = "AT RISK" if y_test[i] == 1 else "SAFE"
    predicted = "AT RISK" if y_pred_test[i] == 1 else "SAFE"
    risk_score = y_pred_proba_test[i]
    confidence = max(risk_score, 1 - risk_score)
    
    emoji = "✓" if (y_test[i] == y_pred_test[i]) else "✗"
    
    print(f"   {student_id:<12} {actual:<8} {predicted:<12} {risk_score:<12.1%} {confidence:<12.1%} {emoji}")

print("   " + "─" * 75 + "\n")

# ============================================================================
# STEP 11: EXPORT RESULTS
# ============================================================================
print("[11] Exporting detailed test results...")
results_df = pd.DataFrame({
    'student_id': df.iloc[idx_test]['student_id'].values,
    'actual_label': ['AT RISK' if y == 1 else 'SAFE' for y in y_test],
    'predicted_label': ['AT RISK' if y == 1 else 'SAFE' for y in y_pred_test],
    'risk_score': y_pred_proba_test,
    'confidence': np.maximum(y_pred_proba_test, 1 - y_pred_proba_test),
    'correct': (y_test == y_pred_test).astype(int)
})

output_file = 'test_results_proper.csv'
results_df.to_csv(output_file, index=False)
print(f"   ✓ Results saved to '{output_file}'\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("✅ VALIDATION COMPLETE")
print("=" * 80)
print(f"\n📊 MODEL PERFORMANCE:")
print(f"   ROC-AUC:    {auc:.2%} {'✅ EXCELLENT' if auc > 0.95 else '⚠️  GOOD' if auc > 0.90 else '❌ NEEDS WORK'}")
print(f"   Recall:     {rec:.2%} {'✅ PERFECT' if rec > 0.95 else '⚠️  GOOD' if rec > 0.85 else '❌ NEEDS WORK'}")
print(f"   Precision:  {prec:.2%}")
print(f"   F1-Score:   {f1:.2%}\n")

# Show error analysis
print(f"📈 ERROR ANALYSIS:")
print(f"   False Negatives: {fn} (students flagged as SAFE but are AT RISK)")
print(f"   False Positives: {fp} (students flagged as AT RISK but are SAFE)\n")

if fn > 0:
    print(f"   ⚠️  WARNING: Model missed {fn} at-risk students (Recall < 100%)")
    print(f"       Consider lowering the threshold or adjusting model weights\n")

print(f"📝 FILES GENERATED:")
print(f"   - {output_file} (detailed predictions)")
print("\n")
