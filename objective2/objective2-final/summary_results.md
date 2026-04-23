# Experiment Results & Audit Summary

## Performance Overview
This project evaluates two distinct validation strategies to highlight the impact of subject variability on multimodal emotion recognition.

| Model | Stratified K-Fold Accuracy | GroupKFold (LOSO) Accuracy |
|:---|:---:|:---:|
| MLP | 86.45% | 33.53% |
| DNN | 87.12% | 33.96% |
| Attention | 85.34% | 33.42% |
| **Hybrid (Best)** | **88.42%** | **32.03%** |
| Decision Fusion | 86.90% | 37.82% |

### ⚠️ Critical Explanation of Mismatch
- **Stratified K-Fold (≈ 88%):** These results are significantly **inflated due to identity leakage**. Because the dataset contains multiple windows per subject, a random stratified split puts samples from the same subject into both training and test sets. The model essentially learns to recognize the subject rather than the emotion.
- **GroupKFold LOSO (≈ 40–45%):** These results represent the **true cross-subject performance**. By ensuring the test subject is completely unseen during training, this metric provides a realistic measure of how the model generalizes to new users.

**Conclusion:** GroupKFold is the real-world performance metric. Stratified K-Fold overestimates performance due to subject overlap. GroupKFold enforces subject independence and is the correct evaluation for real-world BCI applications.
