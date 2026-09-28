# Predicting Regents Diploma Attainment with Decision Trees & Random Forests

**Author:** Nicolette Mtisi  
*Originally completed as a team project for DAV 6150. This repository is my copy of the work.*  
**Tools:** Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## Overview
This project compares **Decision Tree** and **Random Forest** classifiers for predicting how often different student subgroups across New York State school districts earn a **Regents diploma**. The data is New York State high school graduation data for 2018–2019, with more than 73,000 subgroup/district records. I engineered a three-class target, `reg_pct_level` ("low", "medium", "high"), from the continuous Regents diploma percentage using the median as a reference point.

## Workflow
1. **EDA:** univariate, bivariate and multivariate analysis of categorical and numeric variables.
2. **Data preparation:** creating the `reg_pct_level` target, then dropping `reg_pct` and `reg_cnt` to prevent **data leakage**.
3. **Feature selection** with two approaches:
   - **Filter-based:** Mutual Information
   - **Embedded:** Random Forest feature importance
4. **Modeling:** 2 Decision Trees and 2 Random Forests, using `class_weight='balanced'` because about 81% of rows are "medium". Each was evaluated with 5-fold stratified cross-validation.
5. **Model selection and testing** on a stratified holdout set.

## Results
| Model | Feature Selection | CV Accuracy | CV F1 (weighted) |
|-------|-------------------|------------:|-----------------:|
| Decision Tree 1 | Filter-based | 0.665 | 0.707 |
| Decision Tree 2 | Tree-based | 0.669 | 0.712 |
| Random Forest 1 | Filter-based | 0.791 | 0.811 |
| **Random Forest 2** | **Tree-based** | **0.825** | **0.839** |

**Selected model: Random Forest 2**, on the holdout test set:
- Accuracy **83.7%**, weighted F1 **0.850**, weighted precision **0.881**
- Recall of **79%** for "low" and **83%** for "high", the rare, extreme-outcome classes that matter most

## Key Findings
- **The ensemble clearly beats a single tree.** Random Forests gained more than 12 points of F1 and generalized better.
- **Graduation percentage** was the strongest predictor, followed by dropout percentage, enrollment, district need category (`nrc_desc`) and student subgroup.
- **Embedded (tree-based) feature selection** did slightly better than the model-agnostic filter method.
- Test accuracy (83.7%) was close to the CV estimate (82.5%), so the model generalizes reliably.

## Files
| File | Description |
|------|-------------|
| `regents_diploma_trees_forests.ipynb` | Full analysis notebook |
| `M11_Data.csv` | NYS graduation dataset (2018–2019) |

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook regents_diploma_trees_forests.ipynb
```
