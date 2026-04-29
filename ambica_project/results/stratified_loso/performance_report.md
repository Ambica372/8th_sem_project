# Leave-One-Subject-Out (LOSO) Cross-Validation Report

## Overview
This report summarizes the performance of five deep learning models trained on fused EEG and Eye-tracking data. The evaluation strategy uses a **Leave-One-Subject-Out (LOSO)** approach combined with an inner Stratified K-Fold. 

*Note: This report reflects partial pipeline execution (Subjects 0 through 7).*

## Model Performance Summary

The results across the first 8 subjects show that the overall accuracy is relatively low, indicating that the models are struggling to generalize across different subjects. 

The **Decision Fusion** model performed the best but still only achieved an average accuracy of `50.88%`. The individual models hovered around `41%` to `43%` accuracy.

| Model | Mean Accuracy | Standard Deviation |
| :--- | :--- | :--- |
| **Decision Fusion** | **50.88%** | ± 3.99% |
| **Hybrid** | 43.91% | ± 8.27% |
| **DNN** | 43.72% | ± 6.17% |
| **MLP** | 43.57% | ± 7.41% |
| **Attention** | 41.46% | ± 7.82% |

## Detailed Breakdown by Trial (Subject)

The table below shows the exact test accuracy achieved when each subject was held out as the unseen test set. 

| Subject ID | Attention | DNN | Decision Fusion | Hybrid | MLP |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 51.02% | 47.15% | 52.69% | 51.82% | 49.62% |
| **1** | 40.12% | 52.93% | 54.13% | 44.87% | 36.21% |
| **2** | 43.03% | 44.75% | 51.02% | 45.75% | 45.35% |
| **3** | 48.62% | 41.20% | 46.75% | 48.58% | 45.91% |
| **4** | 45.11% | 49.02% | 52.10% | 51.86% | 55.85% |
| **5** | 27.39% | 32.89% | 45.39% | 28.10% | 33.09% |
| **6** | 43.07% | 41.72% | 57.17% | 45.07% | 43.87% |
| **7** | 33.29% | 40.12% | 47.82% | 35.21% | 38.68% |

## Observations & Conclusion

1. **Poor Generalization:** Across all individual models (MLP, DNN, Attention, Hybrid), accuracies drop as low as ~27% for certain subjects (e.g., Subject 5). This indicates high subject variability and poor cross-subject generalization.
2. **High Variance:** The high standard deviation (up to ±8.27% for Hybrid) shows that the models are highly sensitive to which subject is left out.
3. **Fusion Advantage:** The Decision Fusion network consistently outperforms the individual networks (hitting 57% on Subject 6), demonstrating that combining multi-modal outputs improves stability. However, the absolute accuracy remains suboptimal. 

*Generated from `subject_results.csv` and `summary_results_partial.csv`.*
