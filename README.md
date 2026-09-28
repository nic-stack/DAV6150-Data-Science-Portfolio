# Data Science Portfolio: Machine Learning & Predictive Analytics

**Author:** Nicolette Mtisi

These eleven end-to-end data science projects, including a final capstone project, were completed for **DAV 6150 (Data Science)** in the M.S. in Data Analytics & Visualization program at the Katz School of Science and Health, Yeshiva University. They cover the full machine learning workflow, from cleaning messy data to building, tuning and comparing supervised and unsupervised models, and each one ends with business-focused conclusions.

*These projects were originally completed as team assignments. This repository is my copy of the work.*

## Projects

| # | Project | Techniques | Headline Result |
|---|---------|-----------|-----------------|
| 01 | [Cleaning Messy Data: Wine Sales](01-data-cleaning-wine) | EDA, imputation, outlier handling, log transforms | AcidIndex skewness cut from 1.65 to 0.25; dataset fully ML-ready |
| 02 | [Feature Selection: Online News Popularity](02-feature-selection-news-popularity) | Variance/correlation filters, VIF, Lasso, stepwise, PCA, OLS | Stepwise OLS matched the best test R² with 32 of 60 features |
| 03 | [Classification Metrics from Scratch](03-classification-metrics) | Confusion matrix, precision/recall/F1, custom ROC and AUC | Manual AUC (0.850) matched scikit-learn exactly |
| 04 | [Predicting Student Dropouts (NYS)](04-regression-student-dropouts) | Linear, Poisson and Negative Binomial regression | Poisson model: R² = 0.93, RMSE = 12.19 |
| 05 | [Insurance Cross-Sell: Logistic Regression](05-logistic-regression-insurance) | L1/L2 and class-balanced logistic regression, cross-validation | Test ROC-AUC = 0.827 |
| 06 | [Insurance Cross-Sell: KNN & SVM](06-knn-svm-insurance) | KNN (Euclidean/Manhattan), SVM (linear/RBF), PCA, GridSearchCV | KNN ROC-AUC = 0.876; recall +20% over logistic regression |
| 07 | [Online Shopper Purchases: Clustering + SVM](07-clustering-svm-online-shoppers) | Hierarchical clustering, K-Means, Random Forest feature selection, SVM | 90.4% accuracy, ROC-AUC = 0.886 |
| 08 | [Regents Diploma Attainment: Trees & Forests](08-decision-trees-random-forests) | Decision Trees, Random Forests, Mutual Information, class weighting | Random Forest: 83.7% accuracy, weighted F1 = 0.850 |
| 09 | [School Dropout Risk: Gradient Boosting](09-gradient-boosting-dropout-risk) | Decision Tree, Random Forest, Gradient Boosting, SGD, XGBoost, ANOVA | XGBoost: 74.3% accuracy; 88% recall on high-risk districts |
| 10 | [News Popularity: Neural Networks](10-neural-networks-news-popularity) | Feed-forward neural networks (TensorFlow/Keras), dropout, activation/optimizer tuning | 3 architectures compared; best model balances F1 and viral-article recall |
| 11 | ⭐ [**Final Project:** Household Financial Vulnerability](11-final-project-financial-vulnerability) | BLS survey data; XGBoost, SVM, MLP, ensembles, regression, K-Means | Classification PR-AUC = 0.952; expenditure R² = 0.968 |

## Skills Demonstrated
- **Data wrangling:** handling missing, invalid and skewed data, encoding, scaling and transformations (log, Yeo-Johnson)
- **Exploratory analysis:** univariate, bivariate and multivariate EDA with clear visualizations
- **Feature engineering and selection:** VIF, Lasso, stepwise selection, Mutual Information, ANOVA, Random Forest importance, PCA
- **Modeling:** linear, GLM (Poisson and Negative Binomial), logistic regression, KNN, SVM, Decision Trees, Random Forests, Gradient Boosting, XGBoost, neural networks, voting and stacking ensembles, K-Means, hierarchical clustering
- **Evaluation:** cross-validation, hyperparameter tuning, ROC-AUC, precision-recall trade-offs, residual diagnostics
- **Communication:** turning model results into actionable business recommendations

## Tech Stack
Python · pandas · NumPy · scikit-learn · XGBoost · TensorFlow/Keras · statsmodels · SciPy · imbalanced-learn · Matplotlib · Seaborn · Jupyter / Google Colab

## Running the Notebooks
```bash
git clone https://github.com/nic-stack/DAV6150-Data-Science-Portfolio.git
cd DAV6150-Data-Science-Portfolio
pip install pandas numpy scipy scikit-learn xgboost tensorflow statsmodels imbalanced-learn matplotlib seaborn jupyter
jupyter notebook
```
Each project folder has its own README with details, results and the dataset used.
