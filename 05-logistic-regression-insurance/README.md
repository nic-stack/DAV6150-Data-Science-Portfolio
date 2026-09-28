# Predicting Insurance Cross-Sell with Logistic Regression

**Author:** Nicolette Mtisi  
*Originally completed as a team project for DAV 6150. This repository is my copy of the work.*  
**Tools:** Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## Overview
An insurance company wants to know **which existing customers are likely to buy an additional product**. This project builds binary logistic regression models to predict that outcome. The Kaggle dataset has 14,016 customer records, a binary `TARGET`, and 14 explanatory variables covering demographics, loyalty, relationship length, and past product purchases and turnover.

## Workflow
1. **EDA:** univariate analysis of all 15 variables, bivariate analysis against the target (loyalty, product ownership, age, turnover), and a multivariate correlation analysis.
2. **Data preparation:** handling missing values, outliers and unclassified codes, encoding categorical variables, and transforming skewed features.
3. **Feature engineering and selection** using statistical and model-based methods.
4. **Modeling:** three logistic regression models:
   - **Model A:** baseline, L2 regularization
   - **Model B:** class-balanced, L2 regularization
   - **Model C:** sparse, L1 regularization
5. **Evaluation** with cross-validation plus accuracy, precision, recall, F1 and ROC-AUC, then final testing on unseen data.

## Results
| Model | Test ROC-AUC |
|-------|------------:|
| **Model A (Baseline L2)** | **0.827** |
| Model B (Balanced L2) | 0.803 |
| Model C (L1 Sparse) | 0.769 |

**Selected model: Model A**
- Accuracy **75.1%**, Precision **0.746**, Recall **0.636**, F1 **0.686** (buyer class)
- Correctly identified 956 buyers and 1,674 non-buyers

## Business Takeaways
- The model gives the marketing team an **interpretable, reliable way to rank customers** by how likely they are to buy.
- About 36% of buyers were still missed, so **threshold tuning** or more feature engineering could raise recall for campaigns. This problem is continued in [06: KNN & SVM](../06-knn-svm-insurance).

## Files
| File | Description |
|------|-------------|
| `insurance_logistic_regression.ipynb` | Full analysis notebook |
| `M7_Data.csv` | Insurance customer dataset |

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook insurance_logistic_regression.ipynb
```
