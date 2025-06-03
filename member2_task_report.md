
# Member 2: Baseline Decision Tree & Exploratory Analysis

> **Objective:**  
> 1. Load train/test splits prepared by Member 1.  
> 2. Train a default `DecisionTreeClassifier`.  
> 3. Evaluate its performance on the test set (accuracy, precision, recall, F1, ROC-AUC).  
> 4. Plot and interpret a confusion matrix and ROC curve.  
> 5. Visualize feature importances (top 10).  
> 6. Display a shallow decision tree structure for interpretability.  
> 7. Perform 5-fold cross-validation on training data to assess overfitting.

---

## ✅ Cell 1: Imports & Load Train/Test Splits

### Purpose
Load the preprocessed data splits (`X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`) created by Member 1 to ensure consistency in modeling.

```python
# %% Cell 1: Imports & Load Train/Test Splits

import pickle
import pandas as pd
import numpy as np

# Machine learning utilities
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import cross_val_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Load the previously saved train/test splits
with open("X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("y_train.pkl", "rb") as f:
    y_train = pickle.load(f)
with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

print("✅ Loaded train/test splits successfully:")
print(f"  • X_train: {X_train.shape}")
print(f"  • X_test : {X_test.shape}")
print(f"  • y_train: {y_train.shape}")
print(f"  • y_test : {y_test.shape}")
```

- This cell verifies that the data shapes match expectations (e.g., `(32950, 45)` for `X_train` and `(8238, 45)` for `X_test`).

---

## ✅ Cell 2: Train & Evaluate Default Decision Tree

### Purpose
Train a default decision tree without any hyperparameter tuning to establish a baseline, then evaluate using various performance metrics.

```python
# %% Cell 2: Train & Evaluate Default Decision Tree

# 1. Initialize a default DecisionTreeClassifier
clf_default = DecisionTreeClassifier(random_state=42)

# 2. Fit the classifier on the training set
clf_default.fit(X_train, y_train)

# 3. Make predictions on the test set
y_pred = clf_default.predict(X_test)
y_proba = clf_default.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

# 4. Compute evaluation metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# 5. Print results
print("▶️ Default Decision Tree Performance on Test Set:")
print(f"  • Accuracy : {acc:.4f}")
print(f"  • Precision: {prec:.4f}")
print(f"  • Recall   : {rec:.4f}")
print(f"  • F1 Score : {f1:.4f}")
print(f"  • ROC AUC  : {roc_auc:.4f}")
```

**Key Results (Example)**  
- Accuracy: ~0.8971  
- Precision: ~0.5426  
- Recall: ~0.5485  
- F1 Score: ~0.5456  
- ROC AUC: ~0.7449  

These metrics serve as a baseline before applying pruning or other improvements.

---

## ✅ Cell 3: Confusion Matrix & ROC Curve

### Purpose
Visualize the confusion matrix to understand classification errors and plot the ROC curve for probability-based performance.

```python
# %% Cell 3: Confusion Matrix & ROC Curve

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=['Pred 0', 'Pred 1'],
    yticklabels=['True 0', 'True 1']
)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix: Default Decision Tree")
plt.show()

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"Default Tree (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Default Decision Tree")
plt.legend()
plt.show()
```

#### Interpretation
- **Confusion Matrix**:  
  - True Negative: 6881  
  - False Positive: 429  
  - False Negative: 419  
  - True Positive: 509  
- **ROC Curve**: AUC ≈ 0.745 indicates moderate discriminative power.

---

## ✅ Cell 4: Feature Importance (Top 10)

### Purpose
Identify the most influential features according to the default tree, which helps interpret model behavior.

```python
# %% Cell 4: Feature Importance (Top 10)

# 1. Extract feature importances into a pandas Series
importances = pd.Series(clf_default.feature_importances_, index=X_train.columns)

# 2. Select the top 10 features
top10 = importances.sort_values(ascending=False).head(10)

# 3. Plot a horizontal bar chart
plt.figure(figsize=(8, 6))
top10.plot(kind="barh", color="skyblue")
plt.gca().invert_yaxis()  # Show highest importance at the top
plt.xlabel("Importance Score")
plt.title("Top 10 Feature Importances: Default Decision Tree")
plt.show()

# 4. Print the top 10 feature importances
print("▶️ Top 10 Feature Importances:")
for feature, score in top10.items():
    print(f"  • {feature}: {score:.4f}")
```

**Example Output**  
- `duration`: 0.3507  
- `euribor3m`: 0.2197  
- `age`: 0.0980  
- `campaign`: 0.0340  
- `cons.conf.idx`: 0.0261  
- `pdays`: 0.0232  
- `cons.price.idx`: 0.0228  
- `housing_yes`: 0.0173  
- `day_of_week_mon`: 0.0139  
- `loan_yes`: 0.0123  

These top features guide business insights (e.g., longer call durations strongly predict subscription).

---

## ✅ Cell 5: Visualize a Shallow Decision Tree (max_depth=3)

### Purpose
Create a simplified tree structure (depth = 3) to illustrate key decision nodes in an interpretable format.

```python
# %% Cell 5: Visualize a Shallow Decision Tree (max_depth=3)

from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Train a shallow tree for visualization
clf_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_shallow.fit(X_train, y_train)

# 2. Plot the shallow tree
plt.figure(figsize=(20, 10))
plot_tree(
    clf_shallow, 
    feature_names=X_train.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Decision Tree (max_depth=3)")
plt.show()
```

**Interpretation of Key Splits**  
- Root: `duration <= 524.5`  
- Next level includes splits on `euribor3m`, `cons.conf.idx`, and further sub-splits on `duration` or `cons.price.idx`.  
- Helps visualize how major decisions are made (e.g., if `duration` is high and `euribor3m` is above a threshold, likely “Yes”).

---

## ✅ Cell 6: 5-Fold Cross-Validation on Training Data

### Purpose
Assess how well the default (unpruned) tree generalizes within training data via 5-fold cross-validated accuracy.

```python
# %% Cell 6: 5-Fold Cross-Validation on Training Data

from sklearn.model_selection import cross_val_score

# Perform 5-fold CV accuracy
cv_scores = cross_val_score(
    clf_default, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
)

print("▶️ 5-Fold CV Accuracy on Training Data (Default Tree):")
for i, score in enumerate(cv_scores, start=1):
    print(f"  • Fold {i}: {score:.4f}")
print(f"  • Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

**Example Output**  
- Fold 1: 0.8869  
- Fold 2: 0.8879  
- Fold 3: 0.8822  
- Fold 4: 0.8788  
- Fold 5: 0.8941  
- Mean CV Accuracy: 0.8860 ± 0.0052  

This confirms modest overfitting (training CV accuracy slightly below test accuracy).

---

## 📑 Recap & Accomplishments

**Member 2** has successfully:
1. **Loaded** the prepared train/test splits.  
2. **Trained** a default `DecisionTreeClassifier` and **evaluated** its performance on the test set.  
3. **Visualized** model performance through a confusion matrix and ROC curve.  
4. **Identified** top 10 feature importances to interpret model behavior.  
5. **Created** a shallow visual representation of the tree for interpretability.  
6. **Performed** 5-fold CV on the training data to assess overfitting.

These results serve as the **baseline** for later comparison with pruned and tuned decision trees. Subsequent members will build upon this foundation to optimize model performance and draw deeper business insights.

---

**Next Steps for Member 3:**  
- Load the same splits.  
- Conduct cost-complexity pruning path analysis.  
- Run hyperparameter grid search on `max_depth`, `min_samples_split`, and `ccp_alpha`.  
- Evaluate the pruned model’s performance and compare with the baseline.
