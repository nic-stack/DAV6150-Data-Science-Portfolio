# Classifying School Dropout Risk: Gradient Descent vs. Gradient Boosting

**Author:** Nicolette Mtisi  
*Originally completed as a team project for DAV 6150. This repository is my copy of the work.*  
**Tools:** Python · pandas · NumPy · SciPy · scikit-learn · XGBoost · Matplotlib · Seaborn

## Overview
This project sorts New York State school district/subgroup records into three **dropout-risk levels** ("low", "medium", "high"). It compares standard tree models with **gradient-descent** and **gradient-boosting** algorithms to see which approach works best. It uses the same NYS 2018–2019 graduation dataset as [project 08](../08-decision-trees-random-forests), with a new target, `dropout_pct_level`, built from the dropout percentage.

## Workflow
1. **EDA:** univariate, bivariate and multivariate analysis, including pairplots.
2. **Data preparation:** creating the `dropout_pct_level` target, removing `dropout_pct` and `dropout_cnt` to prevent leakage, and checking that the target is correct.
3. **Feature selection and dimensionality reduction:** **ANOVA** tests for categorical predictors, a **PCA** check, and one-hot encoding. This left 4 strong predictors: `grad_pct`, `reg_pct`, `nrc_desc` and `aggregation_type`.
4. **Modeling:** five classifiers compared with cross-validation:
   - Decision Tree (baseline)
   - Random Forest
   - Gradient Boosting
   - Stochastic Gradient Descent (SGD)
   - **XGBoost**
5. **Choosing the best model** by code, then a final evaluation on the test set.

## Results
| Model | CV Accuracy | CV F1 (weighted) |
|-------|------------:|-----------------:|
| **XGBoost** | **0.741** | **0.739** |
| Random Forest | 0.739 | 0.739 |
| Decision Tree | 0.739 | 0.737 |
| Gradient Boosting | 0.732 | 0.729 |
| SGD Classifier | 0.697 | 0.694 |

**Selected model: XGBoost**, on the unseen test set:
- **Accuracy 74.3%**, which matches the CV estimate, so there is no overfitting
- **Recall 0.88 for "high" dropout risk**: it finds almost 9 in 10 high-risk districts
- Precision 0.83 for "low" risk

## Key Findings
- The tree-based ensembles beat the linear **SGD** model by a wide margin, which shows that the link between academic indicators and dropout risk is **non-linear**.
- XGBoost's high recall on the "high" class makes it a practical **early-warning tool** for directing interventions.
- The "medium" class was the hardest to predict (F1 of about 0.63) because it overlaps with the other two classes. Adding attendance or socioeconomic data could help.

## Files
| File | Description |
|------|-------------|
| `dropout_risk_gradient_boosting.ipynb` | Full analysis notebook |

The notebook loads the dataset from GitHub. The same file is also in [`08-decision-trees-random-forests/M11_Data.csv`](../08-decision-trees-random-forests/M11_Data.csv).

## How to Run
```bash
pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn
jupyter notebook dropout_risk_gradient_boosting.ipynb
```
