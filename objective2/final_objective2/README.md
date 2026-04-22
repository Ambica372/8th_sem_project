# Objective 2 — Multimodal Emotion Classification  
## Stratified K-Fold + LOSO Rotation

---

## 📊 Results

| Model | Accuracy |
|------|---------|
| **MLP** | ~80% |
| **DNN** | ~82% |
| **Attention** | ~85% |
| **Hybrid ⭐** | **~86–88%** |
| **Decision Fusion** | ~81% |

---

## 🧠 Data

- **EEG (PCA reduced)**
- **Eye-tracking features**
- **Combined → `X_fused`**
- **4 classes:** Neutral, Sad, Fear, Happy  

---

## ⚙️ Method

### 🔹 Fusion

- **Feature Fusion:** EEG + Eye combined  

---

### 🔹 Models

- **MLP**
- **DNN**
- **Attention Model**
- **Hybrid Model ⭐**
- **Decision Fusion Model**

---

## 🔧 Training

| Parameter | Value |
|----------|------|
| **Optimizer** | AdamW |
| **Learning Rate** | 5e-5 |
| **Weight Decay** | 1e-4 |
| **Batch Size** | 64 |
| **Epochs** | 60 |

### Regularization

- **Batch Normalization**
- **Dropout (0.3)**
- **Gradient Clipping**
- **Early Stopping**
- **LR Scheduler**
- **Class Weights**

---

## 🧹 Preprocessing

- Removed **NaN / Inf**
- Applied **StandardScaler (per fold)**
  - Fit on training data
  - Transform on test data

---

## 🔁 Validation

### Step 1: LOSO (Subject Split)

- 1 subject → test  
- Remaining → train  

---

### Step 2: Stratified K-Fold

- 5-fold split  
- Applied on training data  

---

### 🔄 Pipeline

1. Hold 1 subject out  
2. Apply Stratified K-Fold  
3. Train and evaluate  
4. Repeat across subjects  

---

## 🧠 Conclusion

- **Hybrid model performs best**
- Multimodal fusion improves performance  
- Pipeline is stable and reliable  

---

## 📁 Outputs

- `cv_fold_results.csv`
- `cv_summary_results.csv`
- Confusion matrices
- Model checkpoints  

---




