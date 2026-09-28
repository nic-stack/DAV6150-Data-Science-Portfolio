# Classification Performance Metrics from Scratch

**Author:** Nicolette Mtisi  
*Originally completed as a team project for DAV 6150. This repository is my copy of the work.*  
**Tools:** Python · pandas · scikit-learn · Matplotlib

## Overview
This project implements the core binary classification metrics **by hand in Python**, with no built-in metric functions, and then checks every result against scikit-learn. The dataset has about 180 scored observations, each with its actual class (`class`), predicted class (`scored.class`, using a 0.5 threshold) and predicted probability (`scored.probability`).

## What Was Built
- A **confusion matrix** from `pandas.crosstab`, with TP, FP, TN and FN extracted from it
- Custom functions for **accuracy, precision, sensitivity (recall), specificity and F1 score**
- A custom **ROC curve generator** that sweeps probability thresholds, and an **AUC calculation** using the trapezoidal rule
- A **validation layer** comparing each manual result with scikit-learn's `confusion_matrix`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_curve` and `auc`

## Results
| Metric | Value |
|--------|------:|
| Accuracy | 80.7% |
| Precision | 84.4% |
| Sensitivity (Recall) | 47.4% |
| Specificity | 96.0% |
| F1 Score | 60.7% |
| AUC | 0.850 |

**Every manual calculation matched scikit-learn exactly**, including the AUC (difference 0.0000).

## Key Takeaways
- The model is **conservative**: it rarely produces false positives (96% specificity) but misses more than half of the actual positives.
- If missing a positive case is costly, lowering the classification threshold would trade some precision for better recall.

## Files
| File | Description |
|------|-------------|
| `classification_metrics_from_scratch.ipynb` | Full analysis notebook |
| `M5_Data.csv` | Scored classification dataset |

## How to Run
```bash
pip install pandas scikit-learn matplotlib
jupyter notebook classification_metrics_from_scratch.ipynb
```
