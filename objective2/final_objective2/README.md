Objective 2 — Multimodal Emotion Classification


Stratified K-Fold + LOSO Rotation



📊 Results




Model
Accuracy




MLP
~80%


DNN
~82%


Attention
~85%


Hybrid ⭐
~86–88%


Decision Fusion
~81%





🧠 Data




EEG (PCA reduced)


Eye-tracking features


Combined → X_fused


4 classes: Neutral, Sad, Fear, Happy





⚙️ Method


🔹 Fusion




Feature Fusion → EEG + Eye combined


Decision Fusion → Separate branches, outputs averaged





🔹 Models




MLP


DNN


Attention Model


Hybrid Model ⭐


Decision Fusion Model





🔧 Training




Parameter
Value




Optimizer
AdamW


Learning Rate
5e-5


Weight Decay
1e-4


Batch Size
64


Epochs
60




Regularization




Batch Normalization


Dropout (0.3)


Gradient Clipping


Early Stopping


LR Scheduler


Class Weights





🧹 Preprocessing




Removed NaN / Inf


Ensured data alignment


Applied StandardScaler (per fold)



Fit on training


Transform test









🔁 Validation (FINAL)


Step 1: LOSO (Subject Split)




1 subject → test


Remaining → train





Step 2: Stratified K-Fold




5-fold split


Applied only on training data





🔄 Pipeline




Hold 1 subject out


Apply Stratified K-Fold on remaining data


Train models


Evaluate


Repeat across subjects





🎯 Key Point




Stratified alone → inflated accuracy


LOSO + Stratified → realistic + stable





🧠 Conclusion




Hybrid model performs best


Multimodal fusion improves accuracy


Final pipeline is stable and reliable





📁 Outputs




cv_fold_results.csv


cv_summary_results.csv


Confusion matrices


Model checkpoints




