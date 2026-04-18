# -*- coding: utf-8 -*-
"""
=============================================================================
Objective 2 — 5-Fold Group Cross-Validation Pipeline
VERSION 2: This version uses GroupKFold (subject-wise splitting, NO leakage)
=============================================================================

SIMPLIFIED PIPELINE FOR STABILITY:
- Simple dual-branch model (EEG & Eye -> Dense(128) -> Concat -> Output)
- Early Stopping (patience=10), ReduceLROnPlateau
- Internal 10% validation split for legitimate stopping
- Light Mixup (alpha=0.2, prob=0.2)
- Reduced TTA (runs=3, noise=0.01)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

torch.backends.cudnn.benchmark = True

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REQUIRED_FILES = ["X_eeg_pca.npy", "X_eye_clean.npy", "y.npy"]
_CANDIDATE_DIRS = [
    os.path.join(_SCRIPT_DIR, "..", "objective2", "processed_data"),
    os.path.join(_SCRIPT_DIR, "..", "stage4_pipeline", "processed_data"),
    os.path.join(_SCRIPT_DIR, "processed_data"),
]

DATA_DIR = None
for _d in _CANDIDATE_DIRS:
    _d = os.path.normpath(os.path.abspath(_d))
    if os.path.isdir(_d) and all(os.path.isfile(os.path.join(_d, f)) for f in _REQUIRED_FILES):
        DATA_DIR = _d
        break

if DATA_DIR is None:
    _searched = [os.path.normpath(os.path.abspath(d)) for d in _CANDIDATE_DIRS]
    raise FileNotFoundError(
        "Could not find data folder containing {}.\nSearched:\n{}".format(
            _REQUIRED_FILES, "\n".join("  - {}".format(p) for p in _searched))
    )

# ---------------------------------------------------------------------------
# Hyper-parameters & Setup
# ---------------------------------------------------------------------------
EPOCHS       = 60
BATCH_SIZE   = 64
LR           = 1e-4
WEIGHT_DECAY = 1e-4
N_SPLITS     = 5
N_SUBJECTS   = 32
PATIENCE     = 10
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# =============================================================================
# UTILS: Light Mixup
# =============================================================================
def mixup_data(eeg, eye, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = eeg.size(0)
    index = torch.randperm(batch_size, device=eeg.device)
    mixed_eeg = lam * eeg + (1 - lam) * eeg[index, :]
    mixed_eye = lam * eye + (1 - lam) * eye[index, :]
    y_a, y_b = y, y[index]
    return mixed_eeg, mixed_eye, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =============================================================================
# MODEL ARCHITECTURE: Simple Dual-Branch Model
# =============================================================================
class SimpleDualBranchModel(nn.Module):
    def __init__(self, eeg_dim, eye_dim, num_classes=4):
        super().__init__()
        
        # EEG Branch
        self.eeg_fc = nn.Linear(eeg_dim, 128)
        self.eeg_bn = nn.BatchNorm1d(128)
        self.eeg_drop = nn.Dropout(0.3)
        
        # Eye Branch
        self.eye_fc = nn.Linear(eye_dim, 128)
        self.eye_bn = nn.BatchNorm1d(128)
        self.eye_drop = nn.Dropout(0.3)
        
        # Fusion Output
        self.fusion_fc = nn.Linear(256, 128)
        self.fusion_bn = nn.BatchNorm1d(128)
        self.fusion_drop = nn.Dropout(0.3)
        
        self.out = nn.Linear(128, num_classes)

    def forward(self, eeg, eye):
        eeg_feat = self.eeg_drop(F.relu(self.eeg_bn(self.eeg_fc(eeg))))
        eye_feat = self.eye_drop(F.relu(self.eye_bn(self.eye_fc(eye))))
        
        fused = torch.cat([eeg_feat, eye_feat], dim=-1)
        fused = self.fusion_drop(F.relu(self.fusion_bn(self.fusion_fc(fused))))
        
        return self.out(fused)

# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    eeg_path = os.path.join(DATA_DIR, "X_eeg_pca.npy")
    eye_path = os.path.join(DATA_DIR, "X_eye_clean.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")
    
    X_eeg = np.load(eeg_path, allow_pickle=True)
    X_eye = np.load(eye_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    
    if type(X_eeg) is not np.ndarray: X_eeg = np.array(X_eeg)
    if type(X_eye) is not np.ndarray: X_eye = np.array(X_eye)
    if type(y) is not np.ndarray: y = np.array(y)

    X_eeg = np.asarray(X_eeg, dtype=np.float32)
    X_eye = np.asarray(X_eye, dtype=np.float32)
    y = np.asarray(y).flatten().astype(np.int64)

    bad = (np.isnan(X_eeg).any(axis=1) | np.isinf(X_eeg).any(axis=1) |
           np.isnan(X_eye).any(axis=1) | np.isinf(X_eye).any(axis=1))
           
    X_eeg, X_eye, y = X_eeg[~bad], X_eye[~bad], y[~bad]

    return X_eeg, X_eye, y

def load_or_create_groups(n_samples):
    subject_id_path = os.path.join(DATA_DIR, "subject_ids.npy")
    if os.path.isfile(subject_id_path):
        groups = np.load(subject_id_path, allow_pickle=True).flatten().astype(np.int64)
        if len(groups) > n_samples: groups = groups[:n_samples]
    else:
        groups = np.repeat(np.arange(N_SUBJECTS),
                           int(np.ceil(n_samples / N_SUBJECTS)))[:n_samples]
    return groups

def scale_fold(X_tr, X_te):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)

# =============================================================================
# MAIN LOOP
# =============================================================================
def main():
    X_eeg, X_eye, y = load_data()
    eeg_dim = X_eeg.shape[1]
    eye_dim = X_eye.shape[1]
    n_samples = len(y)
    
    groups = load_or_create_groups(n_samples)
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    fold_accs = []
    
    for fold_k, (train_idx, test_idx) in enumerate(gkf.split(X_eeg, y, groups), start=1):
        print(f"\nTraining Fold {fold_k}...")
        
        Xe_tr_raw, Xey_tr_raw, y_tr = X_eeg[train_idx], X_eye[train_idx], y[train_idx]
        Xe_te_raw, Xey_te_raw, y_te = X_eeg[test_idx], X_eye[test_idx], y[test_idx]

        # Splitting 10% from the training set as validation
        (Xe_t_raw, Xe_v_raw, Xey_t_raw, Xey_v_raw, y_t, y_v) = train_test_split(
            Xe_tr_raw, Xey_tr_raw, y_tr, test_size=0.1, stratify=y_tr, random_state=RANDOM_STATE
        )

        Xe_tr, Xe_te = scale_fold(Xe_t_raw, Xe_te_raw)
        _,     Xe_val = scale_fold(Xe_t_raw, Xe_v_raw)
        
        Xey_tr, Xey_te = scale_fold(Xey_t_raw, Xey_te_raw)
        _,      Xey_val = scale_fold(Xey_t_raw, Xey_v_raw)

        Xe_tr_pt = torch.FloatTensor(Xe_tr)
        Xey_tr_pt = torch.FloatTensor(Xey_tr)
        ytr_pt = torch.LongTensor(y_t)
        
        Xe_val_pt = torch.FloatTensor(Xe_val)
        Xey_val_pt = torch.FloatTensor(Xey_val)
        yval_pt = torch.LongTensor(y_v)
        
        Xe_te_pt = torch.FloatTensor(Xe_te)
        Xey_te_pt = torch.FloatTensor(Xey_te)
        yte_pt = torch.LongTensor(y_te)

        model = SimpleDualBranchModel(eeg_dim, eye_dim)

        classes = np.unique(y_t)
        cw = compute_class_weight("balanced", classes=classes, y=y_t)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw))
        
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(ytr_pt))
            optimizer.zero_grad()
            
            for i in range(0, len(ytr_pt), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                
                # Light MixUp (low probability, low alpha)
                if np.random.rand() < 0.2:
                    mix_eeg, mix_eye, y_a, y_b, lam = mixup_data(Xe_tr_pt[idx], Xey_tr_pt[idx], ytr_pt[idx], alpha=0.2)
                    out = model(mix_eeg, mix_eye)
                    loss = mixup_criterion(criterion, out, y_a, y_b, lam)
                else:
                    out = model(Xe_tr_pt[idx], Xey_tr_pt[idx])
                    loss = criterion(out, ytr_pt[idx])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Validation Step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                out_val = model(Xe_val_pt, Xey_val_pt)
                val_loss = criterion(out_val, yval_pt).item()
                
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"    Early stopping at epoch {ep+1} (val_loss={val_loss:.4f})")
                    break

        # Restore best model for testing
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        
        # Lightweight Test-Time Augmentation (TTA=3)
        with torch.no_grad():
            preds = []
            N_TTA = 3
            for _ in range(N_TTA):
                noise_eeg = torch.randn_like(Xe_te_pt) * 0.01
                noise_eye = torch.randn_like(Xey_te_pt) * 0.01
                out = model(Xe_te_pt + noise_eeg, Xey_te_pt + noise_eye)
                preds.append(F.softmax(out, dim=1))
                
            avg_preds = torch.stack(preds).mean(dim=0)
            _, test_pred = torch.max(avg_preds, 1)
            test_acc = accuracy_score(yte_pt.numpy(), test_pred.numpy())
            
        fold_accs.append(test_acc)
        print("Fold {}: {:.2f}%".format(fold_k, test_acc * 100))

    mean_acc = np.mean(fold_accs)
    print("\nFinal Mean Accuracy: {:.2f}%".format(mean_acc * 100))

if __name__ == "__main__":
    main()
