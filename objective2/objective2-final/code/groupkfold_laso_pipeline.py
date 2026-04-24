# 🎯 Objective 2 — Multimodal Emotion Recognition

## 📌 Overview
This project evaluates whether combining EEG and eye-tracking features improves emotion classification using deep learning.

Two evaluation setups:
- Stratified 5-Fold (window-level) → performance benchmarking  
- GroupKFold (LOSO, subject-level) → cross-subject generalization  

## 📊 Dataset
- Samples: ~37,500  
- Subjects: 15  
- EEG features: 29  
- Eye features: 29  
- Fused features: 58  
- Labels: Emotion classes  

## 🧠 Models
- MLP  
- DNN  
- Attention  
- Hybrid (DNN + Attention)  
- Decision Fusion  

## 🔀 Fusion
Feature Fusion:
- EEG + Eye concatenated → single model  

Decision Fusion:
- Separate models → outputs averaged  

## ⚙️ Training
- Optimizer: AdamW  
- LR: 5e-5  
- Batch size: 64  
- Dropout: 0.3  
- BatchNorm: Yes  
- Early stopping: Yes  
- Scheduler: Yes  
- Class weights: Yes  

## 📊 Evaluation

Stratified 5-Fold:
- Window-level split  
- Same subject can appear in train/test  
- Accuracy: ~80–88%  

GroupKFold (LOSO):
- 1 subject = test  
- No subject overlap  
- Accuracy: ~34%  

## 📈 Results

| Model            | Stratified | LOSO |
|------------------|------------|------|
| MLP              | ~82–85%    | ~30–33% |
| DNN              | ~84–87%    | ~33–37% |
| Attention        | ~83–86%    | ~33–36% |
| Hybrid           | ~85–88%    | ~32–35% |
| Decision Fusion  | ~86–88%    | ~36–40% |

## 🔍 Insight
- Fusion improves performance  
- Results differ across evaluation setups  
- Subject-level testing reflects generalization  

## 📁 Structure
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

## ▶️ Run
python code/stratified_laso_pipeline.py  
python code/groupkfold_laso_pipeline.py  

## ✅ Notes
- No data leakage (scaler fit only on train)  
- Same architecture across folds  
- Proper subject separation in LOSO  

## 🚀 Conclusion
Multimodal fusion improves performance.  
Evaluation method affects observed results.  
Subject-level validation is required for generalization.
