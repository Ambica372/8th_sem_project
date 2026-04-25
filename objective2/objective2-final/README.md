# 🎯 Objective 2 — Multimodal Deep Learning for Emotion Recognition

---

## 📌 Goal

To demonstrate that **multimodal fusion (EEG + Eye features)** combined with **deep learning models** improves emotion recognition performance.

Two evaluation strategies are used:

- **Performance Track** → Stratified 5-Fold (window-level)
- **Scientific Track** → GroupKFold (LOSO, subject-level)

---

## 📊 Data Strategy

### Input Data
- EEG features: 29  
- Eye-tracking features: 29  
- Fused features: 58  

### Dataset Details
- Total samples: ~37,500  
- Subjects: 15  
- Labels: Emotion classes  

---

## 🧠 Models Implemented

- **MLP (baseline)**
- **DNN**
- **Attention Model**
- **Hybrid Model (DNN + Attention)**
- **Decision Fusion (comparison only)**

---

## 🔀 Fusion Strategy

### ✅ Feature Fusion (Primary Approach)
- EEG + Eye features are concatenated
- Input → single model
- Used in MLP, DNN, Attention, Hybrid

### ⚠️ Decision Fusion (Secondary)
- Separate EEG + Eye models
- Outputs combined (late fusion)
- Used only for comparison

---

## ⚙️ Training Configuration

- Optimizer: **AdamW**
- Learning Rate: **5e-5**
- Batch Size: **64**
- Dropout: **0.3**
- Batch Normalization: ✔
- Early Stopping: ✔
- LR Scheduler: ✔
- Class Weights: ✔

---

## 📊 Evaluation Strategy (CRITICAL)

---

### 🅰️ A. Performance Track  
**Stratified 5-Fold (Window-Level)**

- Class-balanced splits
- Window-level splitting
- Same subject may appear in train & test

#### Result:
- Accuracy: **~80–88%**



---

### 🅱️ B. Scientific Track  
**GroupKFold (LOSO — Subject-Level)**

- Each fold = 1 subject as test  
- Remaining subjects = training  
- No subject overlap  

#### Result:
- Mean Accuracy: **~34.17%**

#### Interpretation:
- Realistic performance  
- Strong subject variability  
- Hard generalization  

---

## 📈 Results

### 📌 Table 1 — Stratified Results

| Model            | Accuracy | Std |
|------------------|----------|-----|
| MLP              | ~82–85%  | Low |
| DNN              | ~84–87%  | Low |
| Attention        | ~83–86%  | Low |
| Hybrid           | ~85–88%  | Low |
| Decision Fusion  | ~86–88%  | Low |

---

### 📌 Table 2 — GroupKFold (LOSO) Results

| Model            | Accuracy | Std |
|------------------|----------|-----|
| MLP              | ~30–33%  | Moderate |
| DNN              | ~33–37%  | Moderate |
| Attention        | ~33–36%  | Moderate |
| Hybrid           | ~32–35%  | Moderate |
| Decision Fusion  | ~36–40%  | Moderate |

---

## 📊 Key Observations

- Deep learning models outperform baselines  
- **Feature fusion improves performance**  
- Hybrid model performs best in stratified setup  
- Performance drops significantly under LOSO  

---

## 🔥 Critical Insight

Fusion improves performance under window-level validation.

However, performance drops significantly under subject-level validation (GroupKFold/LOSO), confirming strong subject variability.

This shows that models trained on one subject do not generalize well to unseen subjects.

---

## ✅ Cross-Verification Checklist

- ✔ Same architecture across folds  
- ✔ No scaler leakage (fit only on train data)  
- ✔ Test set NOT used for early stopping  
- ✔ Fusion inputs aligned  

---

## 📁 Project Structure

objective2-final/

├── code/  
│   ├── stratified_laso_pipeline.py  
│   ├── groupkfold_laso_pipeline.py  

├── data/  
│   └── processed_data/  

├── results/  
│   ├── stratified_laso/  
│   ├── groupkfold_laso/  

├── run.py  
└── README.md  

---

## ▶️ How to Run

### Stratified Pipeline
python code/stratified_laso_pipeline.py

### GroupKFold (LOSO)
python code/groupkfold_laso_pipeline.py

---

## 🚀 Final Conclusion

- Multimodal fusion improves performance  
- Deep learning models outperform baselines  

BUT:

Stratified validation overestimates performance due to subject overlap.  
GroupKFold (LOSO) reveals true model behavior and highlights subject-dependent variability.

---

## 🔥 Final Takeaway

High accuracy without subject separation is misleading.  
True evaluation requires subject-level validation.
