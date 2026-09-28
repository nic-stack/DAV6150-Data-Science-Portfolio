# Insurance Cross-Sell Classification with KNN & SVM

**Author:** Nicolette Mtisi  
*Originally completed as a team project for DAV 6150. This repository is my copy of the work.*  
**Tools:** Python · pandas · NumPy · SciPy · scikit-learn · imbalanced-learn · Matplotlib · Seaborn

## Overview
This project follows on from the [logistic regression analysis](../05-logistic-regression-insurance). It uses the same insurance customer dataset (14,000+ records) to test whether **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)** predict better than logistic regression which customers will buy an additional insurance product.

## Workflow
1. **EDA:** reusable analysis functions for numeric and categorical variables, bivariate and multivariate analysis against the target, and integrity checks (cardinality, duplicates, turnover consistency).
2. **Data preparation:** cleaning, outlier handling, encoding and **feature standardization**, which KNN and SVM both need.
3. **Feature selection:** statistical and model-based ranking, keeping the top 5 predictors.
4. **Modeling** with `GridSearchCV` and stratified k-fold cross-validation:
   - KNN with **Euclidean** distance
   - KNN with **Manhattan** distance
   - SVM with a **linear** kernel
   - SVM with an **RBF kernel + PCA**
5. **Evaluation and comparison** on ROC-AUC, F1, precision, recall and accuracy, including a direct comparison with the logistic regression model.

## Results
**Selected model: KNN (Euclidean, top 5 features)**

| Metric | Logistic Regression | **KNN (Euclidean)** | Change |
|--------|-------------------:|-----------------:|------:|
| ROC-AUC | 0.820 | **0.876** | +6.9% |
| F1 Score | 0.650 | **0.765** | +17.7% |
| Precision | 0.680 | **0.780** | +14.7% |
| Recall | 0.625 | **0.750** | +20.0% |
| Accuracy | 0.785 | **0.874** | +11.3% |

## Key Takeaways
- The distance-based and non-linear models (KNN, RBF-SVM) **beat the linear models**, which suggests a **non-linear relationship** between the key predictors (turnover, age, relationship length) and purchase behavior.
- The biggest gain was in **recall**, so the final model finds far more likely buyers, which matters most for targeted marketing.

## Files
| File | Description |
|------|-------------|
| `insurance_knn_svm.ipynb` | Full analysis notebook |

The notebook loads the dataset directly from GitHub. The same file is also in [`05-logistic-regression-insurance/M7_Data.csv`](../05-logistic-regression-insurance/M7_Data.csv).

## How to Run
```bash
pip install pandas numpy scipy scikit-learn imbalanced-learn matplotlib seaborn
jupyter notebook insurance_knn_svm.ipynb
```
