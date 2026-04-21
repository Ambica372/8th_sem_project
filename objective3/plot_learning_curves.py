import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Paths & Hyperparameters
# ---------------------------------------------------------------------------
OBJ3_DIR = os.path.dirname(os.path.abspath(__file__))
OBJ1_DIR = os.path.join(os.path.dirname(OBJ3_DIR), "objective1")
PLOT_DIR = os.path.join(OBJ3_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

EEG_CSV = os.path.join(OBJ1_DIR, "eeg_features.csv")
EYE_CSV = os.path.join(OBJ1_DIR, "eye_features.csv")

N_SPLITS = 5
EPOCHS = 50  # Increased slightly to show convergence tail
BATCH_SIZE = 64
LR = 1e-4
VAL_FRAC = 0.1
PCA_VAR = 0.95
RANDOM_STATE = 42

META_COLS = ["subject_id", "session_id", "trial_id", "window_id", "emotion_label"]

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DeepDNN(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        return self.out(F.relu(self.fc3(x)))

# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------
def load_data():
    eeg_df = pd.read_csv(EEG_CSV)
    eye_df = pd.read_csv(EYE_CSV)
    
    eeg_feat_cols = [c for c in eeg_df.columns if c not in META_COLS]
    eye_feat_cols = [c for c in eye_df.columns if c not in META_COLS]
    
    X_eeg = eeg_df[eeg_feat_cols].values.astype(np.float64)
    X_eye = eye_df[eye_feat_cols].values.astype(np.float64)
    y = eeg_df["emotion_label"].values.astype(np.int64)
    subjects = eeg_df["subject_id"].values.astype(np.int64)
    
    bad = (np.isnan(X_eeg).any(axis=1) | np.isinf(X_eeg).any(axis=1) |
           np.isnan(X_eye).any(axis=1) | np.isinf(X_eye).any(axis=1))
    return X_eeg[~bad], X_eye[~bad], y[~bad], subjects[~bad]

def preprocess_baseline(X_eeg_tr, X_eye_tr, X_eeg_te, X_eye_te):
    pca = PCA(n_components=PCA_VAR, random_state=RANDOM_STATE)
    eeg_tr_pca = pca.fit_transform(X_eeg_tr)
    eeg_te_pca = pca.transform(X_eeg_te)
    
    eye_mean = np.nanmean(X_eye_tr, axis=0)
    eye_mean = np.nan_to_num(eye_mean, nan=0.0)
    eye_tr_c, eye_te_c = X_eye_tr.copy(), X_eye_te.copy()
    for col in range(X_eye_tr.shape[1]):
        eye_tr_c[np.isnan(eye_tr_c[:, col]), col] = eye_mean[col]
        eye_te_c[np.isnan(eye_te_c[:, col]), col] = eye_mean[col]
        
    X_tr = np.hstack([eeg_tr_pca, eye_tr_c])
    X_te = np.hstack([eeg_te_pca, eye_te_c])
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    return X_tr_s, X_te_s

# ---------------------------------------------------------------------------
# Training with History Tracking
# ---------------------------------------------------------------------------
def train_and_track(X_tr, y_tr, input_dim):
    model = DeepDNN(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    Xt, Xv, yt, yv = train_test_split(
        X_tr, y_tr, test_size=VAL_FRAC, stratify=y_tr, random_state=RANDOM_STATE
    )
    Xt_t = torch.FloatTensor(Xt); yt_t = torch.LongTensor(yt)
    Xv_t = torch.FloatTensor(Xv); yv_t = torch.LongTensor(yv)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"Training on {len(yt)} samples, Validating on {len(yv)} samples for {EPOCHS} epochs...")
    
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(yt_t))
        epoch_train_losses = []
        epoch_train_preds = []
        epoch_train_targets = []
        
        for i in range(0, len(yt_t), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            inputs, targets = Xt_t[idx], yt_t[idx]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
            preds = torch.argmax(outputs, 1)
            epoch_train_preds.extend(preds.numpy())
            epoch_train_targets.extend(targets.numpy())
            
        train_loss = np.mean(epoch_train_losses)
        train_acc = accuracy_score(epoch_train_targets, epoch_train_preds)
        
        model.eval()
        with torch.no_grad():
            outputs_v = model(Xv_t)
            val_loss = criterion(outputs_v, yv_t).item()
            preds_v = torch.argmax(outputs_v, 1)
            val_acc = accuracy_score(yv_t.numpy(), preds_v.numpy())
            
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1:02d}/{EPOCHS} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
                  f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
    return history

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_learning_curves(history, out_path):
    epochs = range(1, EPOCHS + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"DNN Learning Curves (Fold 1 Baseline)", fontsize=14, fontweight="bold")
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], label="Train Loss", color="#4C72B0", lw=2)
    axes[0].plot(epochs, history['val_loss'], label="Val Loss", color="#DD8452", lw=2)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, [a * 100 for a in history['train_acc']], label="Train Accuracy", color="#4C72B0", lw=2)
    axes[1].plot(epochs, [a * 100 for a in history['val_acc']], label="Val Accuracy", color="#55A868", lw=2)
    axes[1].axhline(25, ls="--", color="gray", lw=1, label="Chance (25%)")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].set_ylim(20, 100)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved learning curves to: {out_path}")

def main():
    X_eeg, X_eye, y, subjects = load_data()
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    # We only run Fold 1 for the curve demonstration
    tr_idx, te_idx = next(gkf.split(X_eeg, y, groups=subjects))
    
    Xe_tr, Xe_te = X_eeg[tr_idx], X_eeg[te_idx]
    Ey_tr, Ey_te = X_eye[tr_idx], X_eye[te_idx]
    y_tr = y[tr_idx]
    
    print("Preprocessing Fold 1 Baseline...")
    X_tr_s, _ = preprocess_baseline(Xe_tr, Ey_tr, Xe_te, Ey_te)
    
    history = train_and_track(X_tr_s, y_tr, X_tr_s.shape[1])
    
    out_file = os.path.join(PLOT_DIR, "dnn_learning_curves.png")
    plot_learning_curves(history, out_file)
    
if __name__ == "__main__":
    main()
