
# Predictive Modeling Project

This repository contains all the materials and analyses developed for a predictive modeling task, structured into two main use cases:

- **Use Case 1**: Predicting the **Sale Price** of a property.
- **Use Case 2**: Predicting the **Sale Type** of a property.

All work is contained in the `Predictive Modeling >> Answers - Soumiya - RAZZOUK` folder.

---

## Repository Structure

```
Predictive Modeling
└── Answers - Soumiya - RAZZOUK
    ├── Notebooks
    │   ├── Data Exploration.ipynb
    │   ├── Data Exploration.pdf        # Exported PDF
    │   ├── Models.ipynb
    │   └── Models.pdf                  # Exported PDF
    ├── Data
    │   ├── Data.xlsx               # Original dataset provided
    │   ├── uc1_dataset.csv         # Cleaned and transformed dataset for Use Case 1
    │   └── uc2_dataset.csv         # Cleaned and transformed dataset for Use Case 2
```

---


## Notebooks

### 1. `Data Exploration.ipynb`

This notebook contains the full data understanding and preparation pipeline.

#### Data Understanding
- Imported necessary packages and loaded the data.
- Explored **numerical columns** using histograms and basic statistics.
- Explored **categorical columns** using count plots and bar charts.

#### Data Preparation
- Checked for **duplicates** and **missing values**, and performed appropriate imputations.
- Detected and treated **outliers**.
- Conducted **correlation analysis** and **feature engineering**:
  - Combined variables where meaningful.
  - Removed redundant or highly correlated variables.
  - Created new features based on domain understanding.
- Used a **correlation heatmap** to remove strongly correlated features.
- Applied **Label Encoding** to ordinal categorical variables.
- Updated heatmap to reflect newly encoded variables.

#### Splitting by Use Cases

After cleaning, the data was split and processed separately for each use case:

---

##### Use Case 1: Predicting `SalePrice`
- Defined the target: `SalePrice`.
- Applied **Target Encoding with K-Fold** on non-ordinal categorical variables.
- Standardized numerical features.
- Exported the final dataset as `uc1_dataset.csv`.

---

##### Use Case 2: Predicting `SaleType`
- Defined the target: `SaleType`.
- Applied **One-Hot Encoding** to non-ordinal categorical variables.
- Encoded the target (`SaleType`) using **Label Encoding**, without assuming any order.
- Exported the dataset as `uc2_dataset.csv`.

---

### 2. `Models.ipynb`

This notebook contains all modeling experiments and performance evaluation.

#### Part 1: Use Case 1 — Predicting `SalePrice`

- Imported `uc1_dataset`.
- Applied **PCA**.
- Evaluated four model variations using **LightGBM**:
  1. Without PCA, without standardizing the target.
  2. With PCA, target not standardized.
  3. Without PCA, with standardized target.
  4. Fine-tuned LightGBM with standardized target and no PCA.
- Compared model performances using:
  - **Root Mean Squared Error (RMSE)**
  - **R² Score**
- Displayed results in a comparison table.
- Plotted **feature importances** for the best model.

---

#### Part 2: Use Case 2 — Predicting `SaleType`

- Imported `uc2_dataset`.

##### Class Imbalance Handling:
- Initial observation: **Imbalanced target classes**.
- Grouped sale types into:
  - `WD` (Standard Sale)
  - `New` (New House)
  - `Other` (all remaining types)

##### Model Trials:
1. **Random Forest with class weights** – poor performance on minority classes.
2. **Random Forest after grouping classes** – still underperformed.
3. **XGBoost Classifier** – better performance overall.
4. Applied **RandomOverSampler** and retrained XGBoost – improved balance.
5. Final model: **XGBoost with hyperparameter tuning** and **cross-validation** to reduce overfitting.

- Compared model performances using **F1 Score** for multi-class evaluation.

---

## Summary of Deliverables

| Component        | Description |
|------------------|-------------|
| `Data.xlsx`      | Original dataset |
| `uc1_dataset.csv`| Cleaned, encoded, PCA-ready dataset for predicting `SalePrice` |
| `uc2_dataset.csv`| Cleaned and encoded dataset for predicting `SaleType` |
| `Data Exploration.ipynb` | Full data preprocessing pipeline |
| `Models.ipynb`   | All models, training, evaluations, and insights |

---

## Requirements

Ensure you have the following Python packages installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `lightgbm`
- `xgboost`
- `imblearn`

You can install them via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost imbalanced-learn
```

---

## Notes

- PCA was only applied in Use Case 1 to improve performance on high-dimensional data.
- Class imbalance was carefully handled in Use Case 2 using grouping and oversampling.
- Feature importance and evaluation metrics are visualized for both use cases to help interpret model behavior.
