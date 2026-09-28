# Feature Selection & Dimensionality Reduction: Online News Popularity

**Author:** Nicolette Mtisi  
*Originally completed as a team project for DAV 6150. This repository is my copy of the work.*  
**Tools:** Python · pandas · NumPy · scikit-learn · statsmodels · Matplotlib · Seaborn

## Overview
This project predicts how many times an online news article will be shared, using the [UCI Online News Popularity dataset](https://archive.ics.uci.edu/dataset/332/online+news+popularity) (39,797 articles, 60 explanatory variables). The aim was to make the model simpler and easier to interpret by applying **feature selection** and **dimensionality reduction**, without losing predictive performance.

## Workflow
1. **EDA:** univariate analysis of all 61 attributes, plus bivariate plots and correlation heatmaps. The heavily right-skewed `shares` target was log-transformed.
2. **Feature selection:**
   - A variance threshold removed near-constant features
   - Correlation with `log_shares` picked the top 20 predictors
   - **VIF filtering** removed multicollinearity, leaving 10 stable features
   - **LassoCV** confirmed which of those features actually predict shares
   - **Backward elimination** and **bidirectional stepwise** selection were also run
3. **Dimensionality reduction:** **PCA** kept about 90% of the variance in 30 components.
4. **Regression modeling:** OLS models were compared using cross-validation and a hold-out test set, followed by residual diagnostics.

## Results
| Model | # Features | Test R² | Test Adj. R² | Test RMSE |
|-------|-----------:|--------:|-------------:|----------:|
| OLS: Stepwise (bidirectional) | 32 | 0.127 | 0.123 | 0.865 |
| OLS: Backward elimination | 45 | 0.127 | 0.122 | 0.865 |
| OLS: PCA (~90% variance) | 30 | 0.098 | 0.095 | 0.879 |

## Key Findings
- **Stepwise selection matched the best accuracy using fewer features** (32 vs. 45) and kept the model interpretable.
- PCA removed multicollinearity but lost some accuracy and interpretability. Its loadings show "content tone" (PC1), "polarity/topic mix" (PC2) and "keyword/topic richness" (PC3) axes.
- Article popularity has a **weak linear signal** (R² of about 0.13). The residual diagnostics suggest that **non-linear or tree-based models** would be a better next step.

## Files
| File | Description |
|------|-------------|
| `news_popularity_feature_selection.ipynb` | Full analysis notebook |
| `M4_Data.csv` | Online News Popularity dataset |

## How to Run
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
jupyter notebook news_popularity_feature_selection.ipynb
```
