"""
================================================================================
DÉTECTION DES ÉTUDIANTS À RISQUE VIA GRAPH NEURAL NETWORKS
Graph Neural Network Architecture for Academic Risk Detection
================================================================================

Auteur: Advanced AI Research
Date: 2026
Niveau: Recherche Master/PhD

CONTEXTE SCIENTIFIQUE
────────────────────────────────────────────────────────────────────────────
Ce projet implémente une architecture Graph Neural Network pour la détection
précoce des étudiants à risque d'échec académique.

FORMULATION MATHÉMATIQUE
────────────────────────────────────────────────────────────────────────────

1. REPRÉSENTATION EN GRAPHE
   - G = (V, E, X) où:
     * V = ensemble de nœuds (étudiants), |V| = N
     * E = ensemble d'arêtes (relations de similarité)
     * X ∈ ℝ^(N×D) = matrice d'attributs des nœuds (features)

2. MATRICES CLÉS
   - A ∈ ℝ^(N×N) = matrice d'adjacence pondérée
   - D ∈ ℝ^(N×N) = matrice de degrés (D_ii = Σ_j A_ij)
   - Ã = D^(-1/2) A D^(-1/2) = normalization symmétrique

3. COUCHE GCN (Graph Convolutional Network)
   H^(l+1) = σ(Ã H^(l) W^(l))
   où:
   - H^(l) ∈ ℝ^(N×D_l) = activation couche l
   - W^(l) ∈ ℝ^(D_l × D_(l+1)) = poids
   - σ = activation ReLU/ELU
   - Ã = D^(-1/2) A D^(-1/2) (renormalization ChebNet)

4. COUCHE ATTENTION (GAT)
   α_ij = softmax_j(LeakyReLU(a^T[W h_i || W h_j]))
   h_i' = σ(Σ_j α_ij W h_j)
   où:
   - α_ij ∈ [0,1] = coefficient d'attention
   - a ∈ ℝ^(2D) = vecteur d'attention apprenable
   - || = concaténation

5. FONCTION DE PERTE AVEC WEIGHTS (Class Imbalance)
   L = -Σ_i [w_pos · y_i · log(ŷ_i) + w_neg · (1-y_i) · log(1-ŷ_i)]
   où:
   - w_pos = N_neg / (N_pos + N_neg) (poids positif)
   - w_neg = N_pos / (N_pos + N_neg) (poids négatif)

6. MÉTRIQUE ROC-AUC
   AUC = ∫_0^1 TPR(FPR) dFPR
   avec validation robuste par validation croisée stratifiée

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    auc
)
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# PyTorch et PyTorch Geometric
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops

# Configuration globale
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"[INFO] Device utilisé: {DEVICE}")
print(f"[INFO] CUDA disponible: {torch.cuda.is_available()}")


# ================================================================================
# SECTION 1: ANALYSE EXPLORATOIRE DES DONNÉES
# ================================================================================

class DataAnalyzer:
    """Analyse statistique et préparation des données"""
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.features = None
        self.target = None
        
    def load_and_describe(self):
        """Charge et décrit le dataset"""
        print("\n" + "="*80)
        print("STATISTIQUES DESCRIPTIVES DU DATASET")
        print("="*80)
        
        print(f"\n📊 Shape du dataset: {self.df.shape}")
        print(f"\n📋 Colonnes: {self.df.columns.tolist()}")
        
        print("\n" + "-"*80)
        print("DISTRIBUTION DE LA VARIABLE CIBLE (risk_label)")
        print("-"*80)
        
        class_dist = self.df['risk_label'].value_counts()
        print(f"\nClasse 0 (Non à risque): {class_dist[0]} ({100*class_dist[0]/len(self.df):.2f}%)")
        print(f"Classe 1 (À risque):     {class_dist[1]} ({100*class_dist[1]/len(self.df):.2f}%)")
        
        # Class imbalance ratio
        imbalance_ratio = class_dist[1] / class_dist[0]
        print(f"\n⚠️  Ratio de déséquilibre: 1:{1/imbalance_ratio:.2f}")
        print(f"    → Approche: Pondération des classes + Focal Loss")
        
        print("\n" + "-"*80)
        print("STATISTIQUES DES FEATURES")
        print("-"*80)
        print(self.df.describe())
        
        print("\n" + "-"*80)
        print("VALEURS MANQUANTES")
        print("-"*80)
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("✅ Aucune valeur manquante")
        else:
            print(missing[missing > 0])
        
        return class_dist
    
    def correlation_analysis(self):
        """Analyse des corrélations"""
        print("\n" + "="*80)
        print("MATRICE DE CORRÉLATION")
        print("="*80)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        print("\nCorrélations avec risk_label:")
        risk_corr = corr_matrix['risk_label'].sort_values(ascending=False)
        print(risk_corr)
        
        # Visualisation
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, cbar_kws={"shrink": 0.8})
        plt.title('Matrice de Corrélation - Features Académiques', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/Users/mac/Desktop/KAMAL/EDUCATION/MSID/S3/TG/Project/new model/01_correlation_matrix.png', dpi=300)
        plt.close()
        
        return corr_matrix
    
    def prepare_features(self):
        """Prépare les features pour le GNN"""
        print("\n" + "="*80)
        print("PRÉPARATION DES FEATURES")
        print("="*80)
        
        # Sélection des features pertinentes
        feature_cols = ['G1', 'G2', 'G3', 'studytime', 'absences', 
                       'failures', 'progression', 'avg_score', 'engagement_score']
        
        X = self.df[feature_cols].values
        y = self.df['risk_label'].values
        
        # Normalisation robuste (résistant aux outliers)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"\n✅ Features sélectionnées: {len(feature_cols)}")
        print(f"   Dimensionnalité des nœuds: D_in = {X_scaled.shape[1]}")
        print(f"   Nombre de nœuds (étudiants): N = {X_scaled.shape[0]}")
        
        self.features = X_scaled
        self.target = y
        
        return X_scaled, y, feature_cols


# ================================================================================
# SECTION 2: CONSTRUCTION DU GRAPHE
# ================================================================================

class GraphConstructor:
    """Construction intelligente du graphe académique
    
    STRATÉGIE DE CONSTRUCTION D'ARÊTES
    ──────────────────────────────────────────────────────────────────────
    
    Méthode: KNN pondéré avec similarité multi-dimensionnelle
    
    1. CALCUL DE LA SIMILITUDE (Cosinus normalisée)
       sim(i, j) = X_i · X_j / (||X_i|| · ||X_j||)
    
    2. DÉTECTION KNN
       Pour chaque nœud i, connecter aux K-plus-proches-voisins
    
    3. PONDÉRATION (Distance Gaussienne)
       w_ij = exp(-d_ij² / σ²)
       où d_ij = √(Σ(x_ik - x_jk)²) (distance euclidienne)
    
    4. SPÉCIFICATION DE σ
       σ = percentile_75(distances) (adaptatif au dataset)
    
    5. SEUILLAGE ADAPTATIF
       Garder uniquement w_ij > threshold_min pour réduire bruit
    """
    
    def __init__(self, features: np.ndarray, target: np.ndarray, k: int = 10):
        self.features = features
        self.target = target
        self.k = k
        self.N = len(features)
        
    def compute_edge_weights(self, method: str = 'gaussian'):
        """
        Calcule les poids des arêtes
        
        Args:
            method: 'gaussian' (défaut) ou 'cosine'
            
        Returns:
            edges (Tensor): [2, num_edges]
            weights (Tensor): [num_edges]
        """
        print("\n" + "="*80)
        print("CONSTRUCTION DU GRAPHE - ANALYSE DÉTAILLÉE")
        print("="*80)
        
        # 1. Calcul des distances euclidiennes
        print("\n[Étape 1] Calcul de la matrice de distances...")
        distances = cdist(self.features, self.features, metric='euclidean')
        np.fill_diagonal(distances, np.inf)  # Éviter auto-boucles
        
        # 2. Sélection KNN
        print(f"[Étape 2] Sélection des K={self.k} plus proches voisins...")
        knn_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        # 3. Pondération Gaussienne
        print("[Étape 3] Pondération Gaussienne des arêtes...")
        sigma = np.percentile(distances[distances != np.inf], 75)
        print(f"   σ calculé (75e percentile): {sigma:.4f}")
        
        edges = []
        weights = []
        
        for i in range(self.N):
            for j in knn_indices[i]:
                d_ij = distances[i, j]
                w_ij = np.exp(-(d_ij**2) / (2 * sigma**2))
                
                # Seuillage adaptatif
                if w_ij > 0.1:  # Threshold minimum pour éliminer bruit
                    edges.append([i, j])
                    weights.append(w_ij)
        
        edges = np.array(edges).T
        weights = np.array(weights)
        
        # Normalisation des poids
        weights = weights / weights.max()
        
        print(f"\n✅ Graphe construit:")
        print(f"   Nombre de nœuds: {self.N}")
        print(f"   Nombre d'arêtes: {len(weights)}")
        print(f"   Densité du graphe: {2*len(weights)/(self.N*(self.N-1)):.4f}")
        print(f"   Degré moyen: {2*len(weights)/self.N:.2f}")
        
        # Statistics
        print(f"\n📊 Statistiques des poids:")
        print(f"   Min: {weights.min():.4f}")
        print(f"   Max: {weights.max():.4f}")
        print(f"   Moy: {weights.mean():.4f}")
        print(f"   Std: {weights.std():.4f}")
        
        return torch.LongTensor(edges), torch.FloatTensor(weights)
    
    def create_pyg_data(self, edges, weights):
        """Crée un objet PyG Data"""
        x = torch.FloatTensor(self.features)
        y = torch.LongTensor(self.target)
        
        data = Data(
            x=x,
            edge_index=edges,
            edge_attr=weights.unsqueeze(-1),  # Poids comme attribut d'arête
            y=y,
            num_nodes=self.N
        )
        
        return data


# ================================================================================
# SECTION 3: ARCHITECTURE GNN AVANCÉE
# ================================================================================

class HybridGNNModel(nn.Module):
    """
    Architecture GNN Hybride : GCN + GAT
    
    JUSTIFICATION THÉORIQUE
    ──────────────────────────────────────────────────────────────────────
    
    Combinaison optimale pour ce problème:
    
    1. GCN pour agrégation globale
       - Capture relations générales de similarité
       - Efficace computationnellement O(|E|)
    
    2. GAT pour attention locale
       - Poids adaptatifs par arête
       - Capture relations critiques spécifiques
       - Interpretabilité via coefficients d'attention
    
    3. Dropout & BatchNorm
       - Prévention du surapprentissage
       - Accélération convergence
       - Robustesse numérique
    
    Architecture:
    ──────────────
    Input (D_in=9)
         ↓
    GCN Layer 1 (D_in → 64, ReLU) + Dropout(0.3)
         ↓
    GAT Layer (64 → 64, 8 heads, ELU) + Dropout(0.3)
         ↓
    GCN Layer 2 (64 → 32, ReLU) + Dropout(0.2)
         ↓
    Output Layer (32 → 1, Sigmoid)
         ↓
    Binary Classification
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 64, 32],
                 num_heads: int = 8, dropout: float = 0.3):
        super(HybridGNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        
        # Couche 1: GCN
        self.gcn1 = GCNConv(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        # Couche 2: GAT (multi-head attention)
        self.gat = GATConv(hidden_dims[0], hidden_dims[1], 
                          heads=num_heads, concat=False, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        
        # Couche 3: GCN
        self.gcn2 = GCNConv(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        
        # Couche de sortie
        self.fc = nn.Linear(hidden_dims[2], 1)
        
        # Dropout et activation
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU(alpha=1.0)
        self.relu = nn.ReLU()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GCN Layer 1
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # GAT Layer avec multi-head attention
        x = self.gat(x, edge_index)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.dropout(x)
        
        # GCN Layer 2
        x = self.gcn2(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x.squeeze()


# ================================================================================
# SECTION 4: GESTION DU CLASS IMBALANCE
# ================================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss pour la gestion du class imbalance
    
    FORMULE MATHÉMATIQUE
    ──────────────────────────────────────────────────────────────────────
    
    Focal Loss = -α_t · (1 - p_t)^γ · log(p_t)
    
    où:
    - α_t = poids de la classe (w_pos ou w_neg)
    - p_t = probabilité prédite
    - γ = facteur d'importance (focusing parameter)
      * γ ∈ [0, 5], typiquement 2
      * γ=0 : BCE standard
      * γ↑ : focus sur exemples difficiles (hard negatives)
    
    Intuition:
    - Exemples faciles (p_t ≈ 1) : perte ≈ 0
    - Exemples durs (p_t ≈ 0.5) : perte maximale
    - Prévient la domination des exemples faciles et nombreux
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets, class_weights=None):
        # BCE avec poids de classe
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Focal term
        p_t = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = self.alpha * focal_weight * bce_loss
        
        if class_weights is not None:
            focal_loss = focal_loss * class_weights
        
        return focal_loss.mean()


class WeightedBCELoss(nn.Module):
    """BCE Loss pondérée par classe"""
    
    def __init__(self, pos_weight: float = 1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        return F.binary_cross_entropy(inputs, targets, 
                                     weight=self.compute_weights(targets),
                                     reduction='mean')
    
    def compute_weights(self, targets):
        weights = torch.where(targets == 1, 
                            torch.tensor(self.pos_weight, device=targets.device),
                            torch.tensor(1.0, device=targets.device))
        return weights


# ================================================================================
# SECTION 5: ENTRAÎNEMENT ET VALIDATION
# ================================================================================

class GNNTrainer:
    """Pipeline d'entraînement complet avec validation rigoureuse"""
    
    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 0.001, weight_decay: float = 5e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), 
                                    lr=learning_rate, weight_decay=weight_decay)
        
        # Schedulers pour améliorer convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, 
            min_lr=1e-6
        )
        
        self.train_losses = []
        self.val_metrics = {'auc': [], 'f1': [], 'loss': []}
        self.best_auc = 0.0
        self.patience_counter = 0
        
    def compute_class_weights(self, y):
        """Calcule les poids de classe"""
        unique, counts = np.unique(y, return_counts=True)
        weight_dict = {u: (1.0 / c) * (sum(counts) / len(counts)) 
                      for u, c in zip(unique, counts)}
        return weight_dict
    
    def train_epoch(self, train_data, loss_fn):
        """Entraîne une époque"""
        self.model.train()
        
        train_data = train_data.to(self.device)
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(train_data)
        targets = train_data.y.float()
        
        # Loss
        loss = loss_fn(out, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, data, loss_fn):
        """Évalue le modèle"""
        self.model.eval()
        
        data = data.to(self.device)
        out = self.model(data)
        targets = data.y.float()
        
        # Loss
        loss = loss_fn(out, targets).item()
        
        # Métriques
        out_cpu = out.cpu().numpy()
        targets_cpu = targets.cpu().numpy()
        
        y_pred = (out_cpu > 0.5).astype(int)
        auc = roc_auc_score(targets_cpu, out_cpu)
        f1 = f1_score(targets_cpu, y_pred)
        
        return {'auc': auc, 'f1': f1, 'loss': loss, 'out': out_cpu, 'y': targets_cpu}
    
    def train(self, train_data, val_data, epochs: int = 100,
              loss_fn_name: str = 'weighted_bce', early_stopping: bool = True):
        """
        Entraîne le modèle
        
        Args:
            train_data: PyG Data object
            val_data: PyG Data object
            epochs: nombre d'épochs
            loss_fn_name: 'weighted_bce' ou 'focal'
            early_stopping: activation de l'early stopping
        """
        
        # Sélection de la fonction de perte
        if loss_fn_name == 'focal':
            loss_fn = FocalLoss(alpha=0.25, gamma=2.0).to(self.device)
        else:
            # Poids de classe
            class_weights = self.compute_class_weights(train_data.y.numpy())
            pos_weight = class_weights[0] / class_weights[1]
            loss_fn = WeightedBCELoss(pos_weight=pos_weight).to(self.device)
        
        print("\n" + "="*80)
        print(f"ENTRAÎNEMENT DU MODÈLE ({loss_fn_name.upper()})")
        print("="*80)
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_data, loss_fn)
            
            # Validation
            val_metrics = self.evaluate(val_data, loss_fn)
            
            # Store
            self.train_losses.append(train_loss)
            self.val_metrics['auc'].append(val_metrics['auc'])
            self.val_metrics['f1'].append(val_metrics['f1'])
            self.val_metrics['loss'].append(val_metrics['loss'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['auc'])
            
            # Early stopping
            if val_metrics['auc'] > self.best_auc:
                self.best_auc = val_metrics['auc']
                self.patience_counter = 0
                # Sauvegarde du meilleur modèle
                torch.save(self.model.state_dict(), 
                          '/Users/mac/Desktop/KAMAL/EDUCATION/MSID/S3/TG/Project/new model/best_model.pt')
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | "
                      f"Val AUC: {val_metrics['auc']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
                      f"Patience: {self.patience_counter}")
            
            if early_stopping and self.patience_counter >= 20:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(
            '/Users/mac/Desktop/KAMAL/EDUCATION/MSID/S3/TG/Project/new model/best_model.pt'))
        
        return self.train_losses, self.val_metrics


# ================================================================================
# SECTION 6: ÉVALUATION ROBUSTE
# ================================================================================

class RobustEvaluator:
    """Évaluation complète et robuste du modèle"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
    @torch.no_grad()
    def predict(self, data):
        """Prédictions"""
        self.model.eval()
        data = data.to(self.device)
        out = self.model(data)
        return out.cpu().numpy().flatten()
    
    def compute_all_metrics(self, y_true, y_pred_proba):
        """Calcule toutes les métriques"""
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'specificity': confusion_matrix(y_true, y_pred).ravel()[0] / (y_true == 0).sum(),
        }
        
        return metrics, y_pred
    
    def print_report(self, y_true, y_pred_proba, set_name: str = "Test"):
        """Rapport d'évaluation détaillé"""
        metrics, y_pred = self.compute_all_metrics(y_true, y_pred_proba)
        
        print("\n" + "="*80)
        print(f"RAPPORT D'ÉVALUATION - {set_name.upper()}")
        print("="*80)
        
        print(f"\n📊 Métriques de Classification:")
        print(f"   Accuracy:    {metrics['accuracy']:.4f}")
        print(f"   Precision:   {metrics['precision']:.4f}  (TP / (TP + FP))")
        print(f"   Recall:      {metrics['recall']:.4f}  (TP / (TP + FN))")
        print(f"   Specificity: {metrics['specificity']:.4f}  (TN / (TN + FP))")
        print(f"   F1-Score:    {metrics['f1']:.4f}  (harmonic mean)")
        print(f"   ROC-AUC:     {metrics['auc']:.4f}  ✨")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n📋 Matrice de Confusion:")
        print(f"   TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
        print(f"   FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")
        
        print(f"\n📝 Classification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Non-risque', 'À risque']))
        
        return metrics
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path: str = None):
        """Trace la courbe ROC"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('Courbe ROC - Détection Étudiants à Risque', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path: str = None):
        """Trace la matrice de confusion"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-risque', 'À risque'],
                   yticklabels=['Non-risque', 'À risque'],
                   cbar_kws={"shrink": 0.8})
        plt.title('Matrice de Confusion', fontsize=14, fontweight='bold')
        plt.ylabel('Vrai Label', fontsize=12, fontweight='bold')
        plt.xlabel('Prédiction', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


# ================================================================================
# SECTION 7: VISUALISATION DES EMBEDDINGS
# ================================================================================

class EmbeddingVisualizer:
    """Visualisation des node embeddings via t-SNE"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
    @torch.no_grad()
    def extract_embeddings(self, data, layer: int = -2):
        """Extrait les embeddings avant la couche de sortie"""
        self.model.eval()
        data = data.to(self.device)
        
        # Forward jusqu'à la couche -2
        x = data.x
        edge_index = data.edge_index
        
        # GCN Layer 1
        x = self.model.gcn1(x, edge_index)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        
        # GAT Layer
        x = self.model.gat(x, edge_index)
        x = self.model.bn2(x)
        x = self.model.elu(x)
        
        # GCN Layer 2
        x = self.model.gcn2(x, edge_index)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        
        return x.cpu().numpy()
    
    def visualize_tsne(self, embeddings, labels, save_path: str = None):
        """Visualise les embeddings en t-SNE"""
        print("\n[INFO] Réduction t-SNE en cours (peut prendre quelques secondes)...")
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, 
                   random_state=SEED, verbose=0)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 9))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=labels, cmap='RdYlGn_r', s=100, alpha=0.7,
                            edgecolors='black', linewidth=0.5)
        
        cbar = plt.colorbar(scatter, label='Risk Label')
        cbar.set_label('Risk Label (0=Safe, 1=At Risk)', fontsize=11)
        
        plt.title('t-SNE Visualization - Node Embeddings', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
        plt.ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        
        return embeddings_2d


# ================================================================================
# SECTION 8: VALIDATION CROISÉE STRATIFIÉE
# ================================================================================

class StratifiedCrossValidator:
    """Validation croisée stratifiée pour évaluation robuste"""
    
    def __init__(self, n_splits: int = 5, random_state: int = SEED):
        self.n_splits = n_splits
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                                   random_state=random_state)
        self.fold_results = []
        
    def run_cv(self, graph_data, model_class, epochs: int = 100, 
              device: torch.device = DEVICE):
        """Exécute la validation croisée"""
        
        print("\n" + "="*80)
        print(f"VALIDATION CROISÉE STRATIFIÉE ({self.n_splits} folds)")
        print("="*80)
        
        indices = np.arange(len(graph_data.y))
        y = graph_data.y.numpy()
        
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(indices, y), 1):
            print(f"\n─── Fold {fold}/{self.n_splits} ───")
            
            # Split données
            train_mask = torch.zeros(len(y), dtype=torch.bool)
            val_mask = torch.zeros(len(y), dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            
            # Créer sous-ensembles
            train_data = Data(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                edge_attr=graph_data.edge_attr,
                y=graph_data.y,
                train_mask=train_mask,
                val_mask=val_mask
            )
            val_data = Data(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                edge_attr=graph_data.edge_attr,
                y=graph_data.y,
                train_mask=train_mask,
                val_mask=val_mask
            )
            
            # Modèle et entraînement
            model = model_class(input_dim=graph_data.x.shape[1]).to(device)
            trainer = GNNTrainer(model, device)
            trainer.train(train_data, val_data, epochs=epochs, 
                         loss_fn_name='weighted_bce', early_stopping=True)
            
            # Évaluation
            evaluator = RobustEvaluator(model, device)
            y_pred = evaluator.predict(val_data)
            y_true = graph_data.y[val_idx].numpy()
            
            metrics, _ = evaluator.compute_all_metrics(y_true, y_pred)
            self.fold_results.append(metrics)
            
            print(f"   Fold AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")
        
        self._print_cv_summary()
    
    def _print_cv_summary(self):
        """Résumé de la validation croisée"""
        print("\n" + "="*80)
        print("RÉSUMÉ VALIDATION CROISÉE")
        print("="*80)
        
        df_results = pd.DataFrame(self.fold_results)
        
        print("\n📊 Moyennes et écarts-types:")
        for col in df_results.columns:
            mean = df_results[col].mean()
            std = df_results[col].std()
            print(f"   {col:12s}: {mean:.4f} ± {std:.4f}")


print("\n✅ Module GNN Student Risk chargé avec succès!")
