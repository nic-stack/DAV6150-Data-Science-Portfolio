# Cleaning "Messy" Data: Wine Sales Dataset

**Author:** Nicolette Mtisi  
*Originally completed as a team project for DAV 6150. This repository is my copy of the work.*  
**Tools:** Python · pandas · NumPy · Matplotlib · Seaborn

## Overview
This project takes a raw dataset of more than 12,700 wine records and makes it ready for machine learning. The data describes each wine's chemical composition (acidity, pH, alcohol, sulphates and so on), how appealing its label is (`LabelAppeal`), and expert ratings (`STARS`). The response variable, `TARGET`, is the number of cases sold. The raw data has invalid values, missing values, outliers and skewed distributions, and all of them had to be fixed before any modeling.

## Workflow
1. **Load and inspect** the dataset's structure, data types and shape.
2. **Exploratory data analysis (EDA)** using descriptive statistics, histograms, skewness plots, boxplots, count plots and a correlation heatmap.
3. **Data preparation:**
   - Set physically impossible negative chemical values to `NaN`, then impute them
   - Impute missing values: median for numeric columns, mode for categorical/ordinal columns
   - Correct unrealistic alcohol values (below 5% or above 20%)
   - Log-transform the heavily skewed `AcidIndex` into a new `AcidIndex_log` feature
4. **Post-cleaning review:** re-run the EDA to check the improvements before and after cleaning.

## Key Findings
- Several chemical attributes contained **invalid negative values**, and about **26% of `STARS` values were missing**.
- `AcidIndex` was strongly right-skewed. The log transform **cut its skewness from about 1.65 to about 0.25**.
- The cleaned dataset has **no missing or invalid values**, and its distributions are more balanced.
- **Perceptual features (`STARS`, `LabelAppeal`) predicted sales more strongly than chemical composition**, so expert ratings and marketing appear to matter more than the chemistry.

## Files
| File | Description |
|------|-------------|
| `wine_data_cleaning.ipynb` | Full analysis notebook |
| `M3_Data.csv` | Raw wine dataset |

## How to Run
```bash
pip install pandas numpy matplotlib seaborn
jupyter notebook wine_data_cleaning.ipynb
```
The notebook loads the dataset from GitHub, so it needs an internet connection. You can also point it at the local `M3_Data.csv`.
