# Predicting Student Dropouts in New York State High Schools

**Author:** Nicolette Mtisi  
*Originally completed as a team project for DAV 6150. This repository is my copy of the work.*  
**Tools:** Python · pandas · NumPy · statsmodels · scikit-learn · Matplotlib · Seaborn

## Overview
This project builds and compares regression models that predict the **number of student dropouts** (`dropout_cnt`) in New York State high schools for the 2018–2019 school year. The data comes from the **New York State Education Department (NYSED)** and has more than 73,000 rows. Each row is a combination of a school district and a student subgroup, with enrollment, graduation, diploma-type and dropout figures.

## Workflow
1. **EDA:** univariate analysis of every attribute, bivariate checks for redundant code/description pairs, and a multivariate look at enrollment, graduation, still-enrolled and dropout counts.
2. **Data preparation and feature selection:** cleaning, removing redundant and leakage-prone columns, and selecting the most significant predictors.
3. **Model development:** six models were built:
   - 2 × **Linear Regression**
   - 2 × **Poisson Regression**
   - 2 × **Negative Binomial Regression**
4. **Evaluation:** RMSE, R², AIC, log-likelihood and dispersion, with residual plots and actual-vs-predicted plots.
5. **Model selection and testing** on a held-out test set.

## Results
**Preferred model: Poisson Regression, Model 1**
- **Test RMSE = 12.19**
- **Test R² = 0.93**
- Dispersion of about 2.4 (mild overdispersion, within acceptable limits)
- Its actual-vs-predicted points sit close to the 45° line

The Negative Binomial model handled overdispersion slightly better, but Poisson-1 gave the best balance of **accuracy, simplicity and interpretability**.

## Why It Matters
Accurate dropout forecasts help administrators and policymakers **find at-risk districts and subgroups early** and target retention interventions where they will help most.

## Files
| File | Description |
|------|-------------|
| `nys_dropout_regression.ipynb` | Full analysis notebook |
| `Project1_Data.csv` | NYSED graduation and dropout dataset (2018–2019) |

## How to Run
```bash
pip install pandas numpy statsmodels scikit-learn matplotlib seaborn
jupyter notebook nys_dropout_regression.ipynb
```
