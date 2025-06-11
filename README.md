# Bank Marketing Campaign Prediction

A machine learning project to predict whether customers will subscribe to a term deposit based on the UCI Bank Marketing dataset. This repository contains data preprocessing, model training, evaluation, and business-impact analysis.

---

## 📋 Overview

The goal is to build and compare multiple classification models to predict customer subscription (`yes`/`no`) in bank marketing campaigns. The project emphasizes:

* Clean and preprocess the dataset.
* Train baseline and advanced models (Decision Tree, Random Forest, Logistic Regression, Gradient Boosting).
* Evaluate models with metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
* Analyze financial impact using cost-benefit calculations.
* Extract interpretable rules for business insights.

## 🚀 Features

* **Data Preprocessing**: Handling missing values, encoding categorical features, feature selection.
* **Modeling**:

  * Baseline Decision Tree (unpruned)
  * Pruned Decision Tree with Cost-Complexity Pruning
  * Random Forest
  * Logistic Regression (tuned)
  * Gradient Boosting (tuned)
* **Evaluation**: Performance comparison table, confusion matrices, ROC curves.
* **Business Impact**: Profit calculation based on campaign cost and customer lifetime value; threshold calibration.
* **Rule Extraction**: Interpret pruned decision tree rules for actionable marketing strategies.

## 🛠️ Technology Stack

* **Programming Language**: Python 3.x
* **Data Handling**: pandas, numpy
* **Modeling & Evaluation**: scikit-learn
* **Visualization**: matplotlib, seaborn
* **Persistence**: pickle or joblib for saving train/test splits and trained models
* **Environment**: Jupyter Notebooks / VS Code with Jupyter extension

## 📂 Repository Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── bank-full.csv             # Raw dataset (downloaded from UCI)
│   ├── X_train.pkl               # Preprocessed train features
│   ├── X_test.pkl                # Preprocessed test features
│   ├── y_train.pkl               # Train labels
│   └── y_test.pkl                # Test labels
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_baseline_model.ipynb
│   ├── 3_pruning_tuning.ipynb
│   ├── new_algorithm.ipynb       # Combined tuning & comparison
│   └── 4_rule_extraction.ipynb
├── requirements.txt              # Python dependencies
└── LICENSE                       # MIT License or chosen license
```

> Note: Adjust paths as needed. The `data/` directory should contain the raw CSV and preprocessed pickles.

## ⚙️ Setup and Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/https://github.com/juliuslaggah/bank-marketing-dt.git
   cd bank-marketing-prediction
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate        # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt` includes:

   ```text
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   joblib
   ```

4. **Download the dataset**:

   * Place `bank-full.csv` (semicolon-separated) under `data/`.
   * Alternatively, update the path in notebooks if stored elsewhere.

## 📑 Usage

### 1. Data Preprocessing (Member 1)

* Open `notebooks/1_data_preprocessing.ipynb`:

  * Load raw CSV.
  * Replace 'unknown' values, impute with mode.
  * Encode categorical features.
  * Drop highly correlated features.
  * Split into train/test and save as pickles in `data/`.

Run the notebook step-by-step. Ensure `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl` appear in `data/`.

### 2. Baseline Model (Member 2)

* Open `notebooks/2_baseline_model.ipynb`:

  * Load preprocessed splits.
  * Train baseline Decision Tree.
  * Evaluate performance and plot confusion matrix & ROC curve.

### 3. Pruning & Hyperparameter Tuning (Member 3)

* Open `notebooks/3_pruning_tuning.ipynb`:

  * Load splits.
  * Compute cost-complexity pruning path.
  * Perform GridSearchCV to tune `max_depth`, `min_samples_split`, `ccp_alpha`.
  * Visualize impurity vs. alpha and ROC-AUC vs. alpha.
  * Train final pruned tree and evaluate.

### 4. Model Comparison & Business Analysis (Member 4)

* Open `notebooks/new_algorithm.ipynb`:

  * Load splits and retrain pruned Decision Tree with best params.
  * Train Random Forest.
  * Tune Logistic Regression via GridSearchCV (`lr_best`).
  * Tune Gradient Boosting via RandomizedSearchCV (`gb_best`).
  * Evaluate all models on the same test set: Accuracy, Precision, Recall, F1, ROC-AUC.
  * Plot performance radar chart and profit bar chart.
  * Extract decision rules from pruned tree for business use.


## 📊 Running the Comparison Code

In `new_algorithm.ipynb`, after training `dt_pruned`, `rf`, `lr_best`, and `gb_best`, add at the end:

```python
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models = {
    "Decision Tree (Pruned)": dt_pruned,
    "Random Forest": rf,
    "Logistic Regression": lr_best,
    "Gradient Boosting": gb_best
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })
comparison_df = pd.DataFrame(results).set_index("Model")
print(comparison_df)
```

## 🔧 Customization

* **Threshold selection**: Adjust probability thresholds for precision/recall trade-offs. Use profit-based threshold tuning.
* **Feature importance & interpretability**: Use SHAP or partial dependence plots in additional notebooks.
* **Hyperparameter tuning**: Modify CV settings or parameter grids for deeper search.

## 📈 Results & Insights

Refer to `ML_Final_Report BankMarketing.pdf` for detailed analysis, plots, and business recommendations.

## 📝 Contributing

* Fork the repository, create a feature branch, and open a pull request. Ensure new code is documented.
* Update notebooks or scripts to reflect any changes in data or model choices.

## ⚖️ License

This project is licensed under the MIT License. See `LICENSE` for details.

---

**Enjoy exploring the Bank Marketing Prediction pipeline!**
