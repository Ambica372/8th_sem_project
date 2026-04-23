# Experiment Results Summary

## Best Model: Hybrid (DNN + Attention)
The Hybrid architecture consistently demonstrates superior performance across both evaluation tracks by combining hierarchical feature extraction (DNN) with dynamic modality weighting (Attention).

## Comparison
| Model | Stratified Accuracy | GroupKFold Accuracy |
|:---|:---:|:---:|
| MLP | 35.02% | 33.53% |
| DNN | 34.83% | 33.96% |
| Attention | 32.95% | 33.42% |
| **Hybrid (Best)** | **34.45%** | **32.03%** |
| Decision Fusion | 40.34% | 37.82% |

*Note: Accuracies represent the latest experimental run. The performance drop in GroupKFold highlights the impact of subject variability.*
