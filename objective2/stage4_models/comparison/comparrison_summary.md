# Model Comparison Summary

## Overview

This section compares the performance of all implemented models under the SAME pipeline.

Pipeline is fixed:
- Window-based input
- Cleaned dataset (NaN/Inf removed)
- StandardScaler applied
- Same train/test split

Only model architecture changes.

---

## Results Table

| Model | Accuracy | Precision | Recall | F1-score |
|------|---------|----------|--------|----------|
| MLP | 76.03% | 0.7667 | 0.7603 | 0.7596 |
| DNN | 90.35% | 0.9048 | 0.9034 | 0.9033 |
| Attention | 68.25% | 0.6839 | 0.6825 | 0.6824 |
| Hybrid ⭐ | *92.84%* | 0.9286 | 0.9284 | 0.9284 |
| Decision Fusion | 50.54% | 0.5106 | 0.5054 | 0.5044 |

---

## Key Observations

### 1. Best Model: Hybrid
- Highest accuracy: *92.84%*
- Balanced predictions across all classes
- Combines dense + attention effectively

---

### 2. DNN performs strongly
- Achieves *90.3%*
- Shows that deeper models capture feature interactions well

---

### 3. MLP baseline
- Lower accuracy (76%)
- Limited capacity → underfitting

---

### 4. Attention model underperforms
- Only 68%
- Indicates simple attention is not sufficient

---

### 5. Decision Fusion fails
- Only 50%
- Splitting EEG and Eye loses information

---

## Critical Comparison

| Approach | Accuracy |
|---------|---------|
| Trial-based | 25% ❌ |
| Window-based | 92.8% ✅ |

👉 Confirms correct design choice

---

## Final Conclusion

- Feature-level fusion > Decision-level fusion
- Hybrid architecture is optimal
- Data preprocessing and representation had the biggest impact

---

## Supporting Files

- model_comparison.csv
- Confusion matrices (per model)
- Classification reports
