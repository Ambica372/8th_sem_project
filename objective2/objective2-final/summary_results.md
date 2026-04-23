# Results Summary

## Table 1 — Stratified Results (Window-level)
*Inflated due to identity leakage (subject overlap).*

| Model | Accuracy |
|:---|:---:|
| MLP | 86.45% |
| DNN | 87.12% |
| Attention | 85.34% |
| **Hybrid** | **88.42%** |
| Decision Fusion | 86.90% |

## Table 2 — GroupKFold Results (Subject-level)
*Realistic cross-subject performance (no overlap).*

| Model | Mean Accuracy | Std Accuracy |
|:---|:---:|:---:|
| MLP | 33.53% | 2.80% |
| DNN | 33.96% | 4.44% |
| Attention | 33.42% | 1.26% |
| **Hybrid** | **32.03%** | **1.52%** |
| Decision Fusion | 37.82% | 2.59% |

---
**Note**: GroupKFold represents the true real-world performance metric for emotion recognition across different individuals.
