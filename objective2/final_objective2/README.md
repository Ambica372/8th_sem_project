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


Fusion




Feature Fusion → EEG + Eye combined


Decision Fusion → Separate branches, outputs averaged




Models




MLP, DNN, Attention, Hybrid, Decision Fusion





🔧 Training




Optimizer: AdamW


LR: 5e-5


Weight Decay: 1e-4


Batch Size: 64


Epochs: 60




Regularization




BatchNorm


Dropout (0.3)


Gradient Clipping


Early Stopping


LR Scheduler


Class Weights





🧹 Preprocessing




Removed NaN / Inf


StandardScaler (per fold, no leakage)





🔁 Validation (FINAL)


Step 1: LOSO




1 subject = test


Remaining = train




Step 2: Stratified K-Fold




Applied on training data (5-fold)




Pipeline




Hold 1 subject out


Train using Stratified K-Fold


Evaluate


Repeat for subjects





🎯 Key Point




Stratified alone → inflated accuracy (same subject in train/test)


LOSO + Stratified → realistic + stable evaluation





🧠 Conclusion




Hybrid model performs best


Multimodal fusion improves accuracy


Final pipeline is stable and reliable





📁 Outputs




Fold results


Summary results


Confusion matrices


Model checkpoints




