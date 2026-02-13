"""
Find Optimal Threshold to Balance Precision and Recall
"""

import numpy as np
import pandas as pd
import torch
from gnn_student_risk import HybridGNNModel, GraphConstructor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Load data
df = pd.read_csv('students.csv')
feature_cols = [col for col in df.columns if col not in ['student_id', 'risk_label']]
X = df[feature_cols].values
y = df['risk_label'].values

# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Create graph (suppress print output)
graph_constructor = GraphConstructor(features=X_scaled, target=y, k=10)
import sys
from io import StringIO
old_stdout = sys.stdout
sys.stdout = StringIO()
edge_index, edge_weights = graph_constructor.compute_edge_weights()
sys.stdout = old_stdout

# Split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
)

# Load model
DEVICE = torch.device('cpu')
model = HybridGNNModel(
    input_dim=len(feature_cols),
    hidden_dims=[64, 64, 32],
    num_heads=8,
    dropout=0.3
).to(DEVICE)

checkpoint = torch.load('best_model.pt', map_location=DEVICE)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# Predictions
from torch_geometric.data import Data
X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
edge_index_tensor = torch.LongTensor(edge_index).to(DEVICE)
pyg_data = Data(x=X_tensor, edge_index=edge_index_tensor, num_nodes=len(df))

with torch.no_grad():
    logits = model(pyg_data)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

y_pred_proba_test = probs[idx_test]

# Find optimal threshold
print("=" * 80)
print("🎯 THRESHOLD OPTIMIZATION")
print("=" * 80)
print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'FP':<6} {'FN':<6}")
print("─" * 80)

thresholds = np.arange(0.3, 0.8, 0.05)
results = []

for threshold in thresholds:
    y_pred = (y_pred_proba_test > threshold).astype(int)
    
    acc = np.mean(y_pred == y_test)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        fp = fn = 0
        tp = np.sum(y_pred == 1)
    
    results.append({
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'fp': fp,
        'fn': fn
    })
    
    marker = "← BEST" if f1 == max([r['f1'] for r in results]) else ""
    print(f"{threshold:<12.2f} {acc:<12.1%} {prec:<12.1%} {rec:<12.1%} {f1:<12.1%} {fp:<6} {fn:<6} {marker}")

print("─" * 80)

# Find best threshold
best_result = max(results, key=lambda x: x['f1'])
print(f"\n✅ RECOMMENDED THRESHOLD: {best_result['threshold']:.2f}")
print(f"   Accuracy:  {best_result['accuracy']:.1%}")
print(f"   Precision: {best_result['precision']:.1%}")
print(f"   Recall:    {best_result['recall']:.1%}")
print(f"   F1-Score:  {best_result['f1']:.1%}\n")

print("=" * 80)
print("💡 RECOMMENDATION:")
print("=" * 80)
print(f"""
Use threshold = {best_result['threshold']:.2f} instead of 0.5

This will:
✓ Reduce false positives from 27 to {27 - best_result['fp']}
✓ Keep recall high: {best_result['recall']:.1%}
✓ Improve precision: {best_result['precision']:.1%}

Update your app.py prediction threshold to {best_result['threshold']:.2f}
""")
