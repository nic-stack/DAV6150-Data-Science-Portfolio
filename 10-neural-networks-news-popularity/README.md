# Classifying News Article Popularity with Neural Networks

**Author:** Nicolette Mtisi  
*Originally completed as a team project for DAV 6150. This repository is my copy of the work.*  
**Tools:** Python · pandas · NumPy · TensorFlow / Keras · scikit-learn · Matplotlib · Seaborn

## Overview
This project designs, trains and compares **feed-forward neural networks** that sort Mashable news articles into three popularity levels ("low", "medium", "high"). It builds on the [feature selection project](../02-feature-selection-news-popularity) and uses the same Online News Popularity dataset (39,797 articles). This time the problem is framed as **classification**, with a new target, `share_level`, derived from the number of shares.

## Workflow
1. **EDA:** analysis of the dataset's structure, word and sentiment features, timing and topic features, with univariate, bivariate and multivariate views.
2. **Data preparation:** building the `share_level` target from the median number of shares, removing `shares` to prevent leakage, cleaning column names, and ordinal encoding.
3. **Feature selection:** multicollinearity checks and choosing a compact, informative set of features.
4. **Modeling:** three neural network architectures:

| Model | Architecture | Parameters |
|-------|--------------|-----------:|
| Model 1 (Baseline) | 2 hidden layers, ReLU | 307 |
| Model 2 (Deep) | 3 hidden layers + dropout | 2,707 |
| Model 3 (Alternative) | [32, 16] tanh + SGD with momentum | 819 |

5. **Evaluation:** training/validation curves, per-class classification reports and confusion matrices, and final testing on unseen data.

## Results (Test Set)
| Model | Accuracy | Weighted F1 | "High" Class Recall |
|-------|---------:|------------:|--------------------:|
| Model 1 | **61.3%** | 0.551 | 29.9% |
| Model 2 | 60.6% | 0.552 | 33.4% |
| **Model 3** ⭐ | 60.5% | **0.552** | **34.5%** |

**Selected model: Model 3.** It found the most high-popularity ("viral") articles and had the smallest gap between training and validation accuracy (0.3%).

## Key Findings
- **Accuracy alone is misleading here.** Always predicting "medium" already gets about 58%, so models were chosen by weighted F1 and high-class recall.
- **A deeper network didn't help.** Model 2 overfit without improving accuracy, so the ceiling of about 60% comes from the **limited signal in the features**, not the network design.
- Keyword quality (`kw_avg_avg`) and topic channel were the strongest predictors of how widely an article is shared.
- **Limitation:** none of the models learned the small "low" class (about 9% of rows). Next steps would be class weighting, SMOTE or a focal loss.

## Files
| File | Description |
|------|-------------|
| `news_popularity_neural_networks.ipynb` | Full analysis notebook |

The notebook loads the dataset from GitHub. The same file is also in [`02-feature-selection-news-popularity/M4_Data.csv`](../02-feature-selection-news-popularity/M4_Data.csv).

## How to Run
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
jupyter notebook news_popularity_neural_networks.ipynb
```
