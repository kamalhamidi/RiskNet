"""
Graph Neural Network for Student Risk Prediction
Simple version - Easy to understand and modify
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SimpleGNN(nn.Module):
    """
    A simple Graph Neural Network with 3 layers
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, dropout=0.3):
        super(SimpleGNN, self).__init__()
        
        # Graph convolutional layers
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)
        self.gc3 = GCNConv(hidden_dim, 32)
        
        # Fully connected layer
        self.fc = nn.Linear(32, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        # Layer 1
        x = self.gc1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.gc2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.gc3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output
        x = self.fc(x)
        return x


class HybridGNNModel(nn.Module):
    """
    A hybrid model using GCN and GAT layers
    """
    def __init__(self, input_dim, hidden_dims=[64, 64, 32], num_heads=8, dropout=0.3):
        super(HybridGNNModel, self).__init__()
        
        # First GCN layer
        self.gc1 = GCNConv(input_dim, hidden_dims[0])
        
        # GAT layer
        self.gat = GATConv(hidden_dims[0], hidden_dims[1], heads=num_heads, concat=False)
        
        # Second GCN layer
        self.gc2 = GCNConv(hidden_dims[1], hidden_dims[2])
        
        # Output layer
        self.fc = nn.Linear(hidden_dims[2], 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        # GCN Layer 1
        x = self.gc1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GAT Layer
        x = self.gat(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN Layer 2
        x = self.gc2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output
        x = self.fc(x)
        return x


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

class GraphConstructor:
    """
    Build a graph based on student similarity (KNN)
    """
    def __init__(self, features, target, k=10):
        self.features = features
        self.target = target
        self.k = k
        self.n_samples = len(features)
    
    def compute_edge_weights(self):
        """
        Build KNN graph with weighted edges
        """
        from scipy.spatial.distance import euclidean
        from scipy.special import softmax
        
        edge_index = []
        edge_weights = []
        
        # For each node, find k nearest neighbors
        for i in range(self.n_samples):
            # Calculate distances to all other nodes
            distances = []
            for j in range(self.n_samples):
                if i != j:
                    dist = euclidean(self.features[i], self.features[j])
                    distances.append((dist, j))
            
            # Sort by distance and take k nearest
            distances.sort()
            neighbors = distances[:self.k]
            
            # Add edges with distance-based weights
            for dist, j in neighbors:
                edge_index.append([i, j])
                # Weight = 1 / (1 + distance^2)
                weight = 1.0 / (1.0 + dist ** 2)
                edge_weights.append(weight)
        
        edge_index = np.array(edge_index).T
        edge_weights = np.array(edge_weights)
        
        print(f"[INFO] Graph created: {self.n_samples} nodes, {len(edge_weights)} edges")
        
        return edge_index, edge_weights


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

class GNNTrainer:
    """
    Train the GNN model
    """
    def __init__(self, model, device='cpu', lr=0.01):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train_epoch(self, data):
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(data)
        
        # Calculate loss
        target = data.y.unsqueeze(-1).float()
        loss = self.criterion(out, target)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data):
        """Evaluate the model"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            probs = torch.sigmoid(out).squeeze().cpu().numpy()
        
        target = data.y.cpu().numpy()
        pred = (probs > 0.5).astype(int)
        
        acc = accuracy_score(target, pred)
        precision = precision_score(target, pred, zero_division=0)
        recall = recall_score(target, pred, zero_division=0)
        f1 = f1_score(target, pred, zero_division=0)
        auc = roc_auc_score(target, probs)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_prepare_data(csv_path):
    """Load and prepare data"""
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col not in ['student_id', 'risk_label']]
    
    X = df[feature_cols].values
    y = df['risk_label'].values
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_cols, scaler


def create_graph_data(features, labels, k=10):
    """Create graph data for PyTorch Geometric"""
    gc = GraphConstructor(features=features, target=labels, k=k)
    edge_index, edge_weights = gc.compute_edge_weights()
    
    X_tensor = torch.FloatTensor(features)
    y_tensor = torch.LongTensor(labels)
    edge_index_tensor = torch.LongTensor(edge_index)
    edge_weight_tensor = torch.FloatTensor(edge_weights)
    
    data = Data(
        x=X_tensor,
        edge_index=edge_index_tensor,
        edge_weight=edge_weight_tensor,
        y=y_tensor,
        num_nodes=len(features)
    )
    
    return data


def predict_single(model, new_student, scaler, feature_cols, device='cpu'):
    """Predict risk for a single student"""
    df_original = pd.read_csv('students.csv')
    X_original = df_original[feature_cols].values
    y_original = df_original['risk_label'].values
    
    X_original_scaled = scaler.transform(X_original)
    df_new = pd.DataFrame([new_student])
    X_new_scaled = scaler.transform(df_new[feature_cols])
    
    # Combine data
    X_combined = np.vstack([X_original_scaled, X_new_scaled])
    
    # Build graph
    gc = GraphConstructor(features=X_combined, target=np.hstack([y_original, [0]]), k=10)
    
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    edge_index, edge_weights = gc.compute_edge_weights()
    sys.stdout = old_stdout
    
    # Make prediction
    X_tensor = torch.FloatTensor(X_combined).to(device)
    edge_index_tensor = torch.LongTensor(edge_index).to(device)
    
    data = Data(
        x=X_tensor,
        edge_index=edge_index_tensor,
        edge_weight=torch.FloatTensor(edge_weights).to(device),
        num_nodes=len(X_combined)
    )
    
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    return probs[-1]


print("✅ GNN modules loaded successfully!")
