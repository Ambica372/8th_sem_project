Stratified vs GroupKFold — Emotion Recognition Evaluation

---

📊 FINAL RESULTS

Method| Mean Accuracy| Best Fold| Worst Fold| Validity
Stratified K-Fold| ~80% – 88%| High| High| ❌ Misleading
Group K-Fold| 40.87%| 46.07%| 32.00%| ✅ Real

---

⚠️ Key Insight

High accuracy from StratifiedKFold is misleading due to data leakage.

GroupKFold provides realistic performance by preventing subject overlap.

---

🧠 Problem in StratifiedKFold

- Splits based on class distribution only
- Same subject appears in:
  - Training set ❌
  - Testing set ❌
- Model memorizes subject-specific patterns

👉 Result: artificially high accuracy

---

✅ Solution: GroupKFold

- Splits based on subject IDs
- Each subject appears in only one fold
- No leakage between train and test

👉 Model is forced to generalize

---

🧪 GroupKFold Performance (Our Final Model)

Fold| Accuracy
Fold 1| 46.07%
Fold 2| 42.58%
Fold 3| 44.73%
Fold 4| 32.00%
Fold 5| 38.98%

Final Mean Accuracy: 40.87%

---

⚙️ Model Improvements Applied

Compared to baseline models, the following enhancements were implemented:

- Multimodal Fusion (EEG + Eye features)
- Residual Learning Blocks
- Attention Mechanism
- Cross-Modal Gating
- MixUp Data Augmentation
- Test-Time Augmentation (TTA)
- Label Smoothing
- AdamW Optimizer
- Learning Rate Scheduling
- Early Stopping

---

⚔️ Stratified vs GroupKFold Comparison

Factor| StratifiedKFold| GroupKFold
Class balance| ✅ Yes| ⚠️ Partial
Subject leakage| ❌ Yes| ✅ No
Real-world validity| ❌ No| ✅ Yes
Accuracy| High (fake)| Lower (real)

---

📁 Project Structure

objective2/
│
├── stratified_vs_groupkfold/
│   ├── README.md
│   ├── stratified_cv_results/
│   ├── groupkfold_cv_results/
│
├── processed_data/
├── stage4_models/

---

▶️ How to Run

Stratified K-Fold

python stage4_cv.py

Group K-Fold (Recommended)

python stage4_cv_groupkfold.py

---

📌 Final Conclusion

- StratifiedKFold produces inflated performance due to leakage
- GroupKFold produces true generalization performance

Accuracy dropped from ~85% → ~40%, but:

«This reflects the model’s actual ability on unseen subjects»

---
