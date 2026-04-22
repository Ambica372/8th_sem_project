🎯 Objective 2 — Multimodal Emotion Classification


EEG + Eye Tracking using Deep Learning



📊 Final Results




Model
Accuracy (Mean ± Std)
F1 Score (Mean ± Std)




MLP
80–81%
~80%


DNN
81–82%
~81%


Attention
85–86%
~85%


Hybrid ⭐
86–88%
~87%


Decision Fusion
~81%
~81%




👉 Best Model: Hybrid (Attention + Dense)



🧠 Problem Statement


Classify human emotions using multimodal physiological signals:




Modality
Description




EEG
Brain activity signals


Eye
Eye-tracking features




Classes (4): Neutral, Sad, Fear, Happy



📁 Dataset




File
Description
Shape




X_fused.npy
EEG + Eye combined
(N, 58)


X_eeg_pca.npy
EEG (PCA reduced)
(N, 29)


X_eye_clean.npy
Eye features
(N, 29)


y.npy
Labels
(N,)




Total samples: ~37,500

Class distribution: Balanced



⚙️ Methodology



🔹 1. Feature Fusion




Fusion Type
Description
Used In




Feature Fusion (Early)
EEG + Eye concatenated
MLP, DNN, Attention, Hybrid


Decision Fusion (Late)
Separate branches, outputs averaged
Decision Fusion Model





🔹 2. Models Implemented




Model
Key Idea




MLP
Baseline fully connected network


DNN
Deeper architecture


Attention
Learns feature importance


Hybrid ⭐
Combines attention + dense


Decision Fusion
Separate EEG & Eye networks





🔹 3. Training Configuration




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





🔹 4. Regularization & Stability




Technique
Purpose




Batch Normalization
Stabilizes training


Dropout (0.3)
Prevents overfitting


Gradient Clipping
Avoids exploding gradients


Early Stopping
Stops over-training


LR Scheduler
Adjusts learning rate dynamically


Class Weights
Handles imbalance





🔹 5. Data Preprocessing




Removed NaN / Inf values


Ensured alignment across:



EEG


Eye


Labels






Applied StandardScaler per fold



Fit only on training data


Applied to test data








👉 Prevents data leakage



🔁 Validation Strategy



✅ Stratified K-Fold (Primary)




Parameter
Value




Splits
5


Type
Window-level


Shuffle
Yes




👉 Maintains class balance

👉 Produces high accuracy (80–88%)



🔁 LOSO-style Subject Rotation




Concept
Description




Split
One subject as test


Train
Remaining subjects


Purpose
Stability check




👉 Checks if model performance is consistent across subjects



⚠️ Important Observation


Why Stratified Accuracy is High




Same subject appears in train & test


Model learns subject-specific patterns




👉 Leads to inflated accuracy



Why LOSO is Important




No subject overlap


Tests true generalization




👉 Accuracy drops but becomes realistic



📈 Performance Interpretation




Metric
Value




Random Baseline
25%


Achieved Accuracy
80–88%




👉 Model learns meaningful patterns

⚠️ But stratified results are optimistic



⚖️ Comparison




Aspect
Stratified K-Fold
LOSO




Split Type
Window-level
Subject-level


Accuracy
High
Lower


Leakage
Present
Removed


Reliability
Medium
High


Real-world validity
❌
✅





📊 Key Insights




Hybrid model performs best consistently


Attention improves feature learning


Multimodal fusion improves accuracy


Regularization stabilizes training


Results are consistent across folds





📁 Outputs Generated




File
Description




cv_fold_results.csv
Fold-wise metrics


cv_summary_results.csv
Final summary


cv_performance_chart.png
Accuracy/F1 comparison


cv_fold_variance.png
Fold stability


.pth files
Model checkpoints


Confusion matrices
Per fold


Reports
Classification reports





🧠 Final Conclusion




Multimodal learning improves emotion classification


Hybrid model achieves best performance (~87%)


Stratified K-Fold gives strong benchmark results


LOSO ensures robustness validation


Pipeline is optimized and stable





🚀 Status


✅ Objective 2 Completed

✅ Models trained and evaluated

✅ Performance optimized

✅ Stability verified


