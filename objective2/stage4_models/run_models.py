import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

ROOT_MODELS = r"c:\Users\swamy\OneDrive\Desktop\8th_sem_new\stage4_models"
ROOT_PREP = r"c:\Users\swamy\OneDrive\Desktop\8th_sem_new\stage4_pipeline\processed_data"
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-4

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        return self.out(x)

class DeepDNN(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(DeepDNN, self).__init__()
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

class AttentionModel(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(AttentionModel, self).__init__()
        self.attention_weights = nn.Linear(input_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, 64)
        self.out = nn.Linear(64, num_classes)
    def forward(self, x):
        attn_scores = torch.sigmoid(self.attention_weights(x))
        return self.out(F.relu(self.fc1(x * attn_scores)))

class HybridModel(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(HybridModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.attention_weights = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        attn_scores = torch.sigmoid(self.attention_weights(x))
        return self.out(F.relu(self.fc2(x * attn_scores)))

class DecisionFusion(nn.Module):
    def __init__(self, eeg_dim, eye_dim, num_classes=4):
        super(DecisionFusion, self).__init__()
        self.eeg_fc = nn.Linear(eeg_dim, 64)
        self.eeg_out = nn.Linear(64, num_classes)
        self.eye_fc = nn.Linear(eye_dim, 32)
        self.eye_out = nn.Linear(32, num_classes)
    def forward(self, eeg_x, eye_x):
        return (self.eeg_out(F.relu(self.eeg_fc(eeg_x))) + self.eye_out(F.relu(self.eye_fc(eye_x)))) / 2.0

def train_eval_model(model_name, model, X_tr, y_tr, X_ts, y_ts, folder_name, X2_tr=None, X2_ts=None):
    model_dir = os.path.join(ROOT_MODELS, folder_name)
    ensure_dir(model_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    y_tr_t, y_ts_t = torch.LongTensor(y_tr), torch.LongTensor(y_ts)
    X_tr_t, X_ts_t = torch.FloatTensor(X_tr), torch.FloatTensor(X_ts)
    
    is_dec = X2_tr is not None
    if is_dec:
        X2_tr_t, X2_ts_t = torch.FloatTensor(X2_tr), torch.FloatTensor(X2_ts)

    print(f"\n--- Training {model_name} ---")
    first_batch = True
    best_acc = 0.0
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(y_tr))
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, len(y_tr), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            
            if first_batch:
                print(f"Batch shape inside training loop: {X_tr_t[idx].shape}")
                if is_dec:
                    print(f"Secondary Batch shape: {X2_tr_t[idx].shape}")
                first_batch = False
                
            optimizer.zero_grad()
            if is_dec:
                out = model(X_tr_t[idx], X2_tr_t[idx])
            else:
                out = model(X_tr_t[idx])
                
            loss = criterion(out, y_tr_t[idx])
            
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected at epoch {ep+1}. STOPPING training immediately.")
                break
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        model.eval()
        with torch.no_grad():
            if is_dec:
                val_preds = model(X_ts_t, X2_ts_t)
            else:
                val_preds = model(X_ts_t)
            _, val_pred_cls = torch.max(val_preds, 1)
            val_acc = accuracy_score(y_ts_t.numpy(), val_pred_cls.numpy())
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
        
        if batches > 0:
            print(f"Epoch [{ep+1}/{EPOCHS}], Loss: {epoch_loss/batches:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print("NaN loss break triggered, breaking outer loop.")
            break

    # Load best model for final evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    model.eval()
    with torch.no_grad():
        if is_dec:
            preds = model(X_ts_t, X2_ts_t)
        else:
            preds = model(X_ts_t)
        _, pred_cls = torch.max(preds, 1)

    np_pred = pred_cls.numpy()
    np_true = y_ts_t.numpy()
    
    unique, counts = np.unique(np_pred, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Final predictions distribution (best model): {dist}")
    
    if len(unique) <= 1:
        print("MODEL COLLAPSE DETECTED: All predictions are the same class.")
    
    acc = accuracy_score(np_true, np_pred)
    cm = confusion_matrix(np_true, np_pred)
    report = classification_report(np_true, np_pred, zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(np_true, np_pred, average='weighted', zero_division=0)

    with open(os.path.join(model_dir, 'accuracy.txt'), 'w') as f:
        f.write(f'Accuracy: {acc:.4f}\n')
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    plt.close()
    
    return acc, p, r, f1, len(np.unique(np_pred)), report, dist

def main():
    X_fused = np.load(os.path.join(ROOT_PREP, 'X_fused.npy'))
    y = np.load(os.path.join(ROOT_PREP, 'y.npy'))
    X_eeg_pca = np.load(os.path.join(ROOT_PREP, 'X_eeg_pca.npy'))
    X_eye_clean = np.load(os.path.join(ROOT_PREP, 'X_eye_clean.npy'))

    import sys
    sys.path.append(os.path.dirname(os.path.dirname(ROOT_PREP)))
    from pipeline.config import EEG_ZIP, EYE_ZIP
    from pipeline.data_loader import load_mat_from_zip, extract_trials

    eeg_mats = load_mat_from_zip(EEG_ZIP)
    eye_mats = load_mat_from_zip(EYE_ZIP)
    
    print("\n--- DATASET SHAPES (NO AGGREGATION) ---")
    print(f"Loaded X_fused shape: {X_fused.shape}")
    print(f"Loaded y shape: {y.shape}")

    # Detect corruption across all feature sets
    nan_fused = np.isnan(X_fused).any(axis=1)
    inf_fused = np.isinf(X_fused).any(axis=1)
    nan_eeg = np.isnan(X_eeg_pca).any(axis=1)
    inf_eeg = np.isinf(X_eeg_pca).any(axis=1)
    nan_eye = np.isnan(X_eye_clean).any(axis=1)
    inf_eye = np.isinf(X_eye_clean).any(axis=1)
    
    bad_rows = nan_fused | inf_fused | nan_eeg | inf_eeg | nan_eye | inf_eye
    corrupted_count = bad_rows.sum()
    
    print(f"\n--- DATA CLEANING ---")
    print(f"Total samples: {len(y)}")
    print(f"Corrupted samples count: {corrupted_count}")
    
    # Remove corrupted rows
    X_fused = X_fused[~bad_rows]
    X_eeg_pca = X_eeg_pca[~bad_rows]
    X_eye_clean = X_eye_clean[~bad_rows]
    y = y[~bad_rows]
    
    # Verify
    assert not np.isnan(X_fused).any()
    assert not np.isinf(X_fused).any()
    assert not np.isnan(X_eeg_pca).any()
    assert not np.isinf(X_eeg_pca).any()
    assert not np.isnan(X_eye_clean).any()
    assert not np.isinf(X_eye_clean).any()

    # VALIDATE INPUT
    if X_fused.shape[1] != 58:
        print(f"ERROR: Input shape validation failed. Expected (N, 58), got {X_fused.shape}.")
        sys.exit(1)
        
    print(f"Validated X_fused shape: {X_fused.shape}")
    print(f"Validated y shape: {y.shape}")

    # SCALING FIX
    global_scaler = StandardScaler()
    X_fused = global_scaler.fit_transform(X_fused)

    X_f_tr, X_f_ts, y_tr, y_ts = train_test_split(X_fused, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)
    X_e_tr, X_e_ts, _, _ = train_test_split(X_eeg_pca, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)
    X_ey_tr, X_ey_ts, _, _ = train_test_split(X_eye_clean, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)
    
    print("\n--- TRAIN/TEST SPLIT SHAPES ---")
    print(f"Train split shapes -> X: {X_f_tr.shape}, y: {y_tr.shape}")
    print(f"Test split shapes  -> X: {X_f_ts.shape}, y: {y_ts.shape}")

    fdim = X_f_tr.shape[1]
    edim = X_e_tr.shape[1]
    eydim = X_ey_tr.shape[1]

    results = []
    
    models_to_test = [
        ('MLP', BaselineMLP(fdim), 'mlp'),
        ('DNN', DeepDNN(fdim), 'dnn'),
        ('Attention', AttentionModel(fdim), 'attention'),
        ('Hybrid', HybridModel(fdim), 'hybrid')
    ]
    
    best_overall_acc = 0.0
    best_overall_model = None
    all_dists = {}

    for n, m, d in models_to_test:
        a, p, r, f1, uni_preds, report_str, dist = train_eval_model(n, m, X_f_tr, y_tr, X_f_ts, y_ts, d)
        results.append([n, a, p, r, f1])
        all_dists[n] = dist
        
        if a > best_overall_acc:
            best_overall_acc = a
            best_overall_model = n

    m_df = DecisionFusion(edim, eydim)
    a, p, r, f1, uni_preds, report_str, dist = train_eval_model('Decision Fusion', m_df, X_e_tr, y_tr, X_e_ts, y_ts, 'decision_fusion', X_ey_tr, X_ey_ts)
    results.append(['Decision Fusion', a, p, r, f1])
    all_dists['Decision Fusion'] = dist
    
    if a > best_overall_acc:
        best_overall_acc = a
        best_overall_model = 'Decision Fusion'

    comp_dir = os.path.join(ROOT_MODELS, 'comparison')
    ensure_dir(comp_dir)
    df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
    df.to_csv(os.path.join(comp_dir, 'model_comparison.csv'), index=False)

    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Best Model: {best_overall_model}")
    print(f"Highest Validation Accuracy: {best_overall_acc*100:.2f}%\n")
    print("Class Balance / Distributions:")
    for model_name, dist in all_dists.items():
        print(f" - {model_name}: {dist}")

    print("\nModel Comparison:")
    print(df.to_string(index=False))
    print("="*50)

if __name__ == '__main__':
    main()
