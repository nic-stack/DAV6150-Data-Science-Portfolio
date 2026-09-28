# Can We Predict Purchases from Website Behavior? Clustering + SVM

**Author:** Nicolette Mtisi  
*Originally completed as a team project for DAV 6150. This repository is my copy of the work.*  
**Tools:** Python · pandas · NumPy · SciPy · scikit-learn · Matplotlib · Seaborn

## Overview
This project uses the [UCI Online Shoppers Purchasing Intention dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) to answer two questions:
1. **Unsupervised:** do website visitors fall into natural behavioral segments?
2. **Supervised:** can we predict whether a session ends in a purchase (`Revenue`)?

The features describe each session: page visits and time spent (administrative, informational and product-related pages), bounce and exit rates, page values, month, visitor type, and traffic source.

## Workflow
1. **Pre-clustering EDA** of every numeric and categorical variable, plus bivariate relationships.
2. **Data preparation:** encoding, **Yeo-Johnson** transformation of skewed features, and **Z-score** scaling.
3. **Clustering:**
   - **Hierarchical clustering** (Ward linkage) with a dendrogram
   - **K-Means**, with the elbow method and silhouette scores used to choose K
   - Post-clustering EDA, and a comparison of the cluster labels with actual purchases
4. **Feature selection** with **Random Forest importance** and **PCA**.
5. **SVM modeling:** several SVMs with different kernels and feature sets, tuned and validated with 5-fold stratified cross-validation.
6. **Final evaluation** on an unseen test set.

## Results
**Clustering:** the dendrogram, elbow plot and silhouette scores all point to **K = 2** segments:
- **High-intent** visitors: high engagement, low exit rates, high `PageValues` (**24.2% purchase rate**)
- **Low-intent** visitors: low engagement, high bounce rates (**6.7% purchase rate**)

The clusters agree with actual purchase labels only 41.4% of the time. They are useful for segmentation but not for direct prediction, which is why a supervised model was needed.

**SVM (Random Forest–selected features), unseen test set:**
| Metric | Value |
|--------|------:|
| Accuracy | **90.4%** |
| ROC-AUC | **0.886** |
| Precision (purchase) | 0.719 |
| Recall (purchase) | 0.623 |
| F1 (purchase) | 0.668 |

The strongest predictors were **`PageValues`**, **`ExitRates`** and **`ProductRelated_Duration`**.

## Business Takeaways
- **Segmentation** supports targeted campaigns: conversion offers for high-intent visitors and re-engagement for low-intent visitors.
- The **SVM model** could run in real time to trigger personalized recommendations or live-chat support.
- Next steps: improve recall (for example with SMOTE or cost-sensitive learning) and try ensemble models.

## Files
| File | Description |
|------|-------------|
| `online_shoppers_clustering_svm.ipynb` | Full analysis notebook |
| `Project2_Data.csv` | Session features |
| `Project2_Data_Labels.csv` | Actual purchase labels (`Revenue`) |

## How to Run
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
jupyter notebook online_shoppers_clustering_svm.ipynb
```
