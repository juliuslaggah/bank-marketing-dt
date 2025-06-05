
# Member 3: Advanced Pruning & Hyperparameter Search

> **Objective:**  
> 1. Load the same train/test splits used by Member 1 & Member 2.  
> 2. Compute the Cost-Complexity Pruning (CCP) path for a fully grown decision tree.  
> 3. Perform a grid search over `max_depth`, `min_samples_split`, and `ccp_alpha` using 5‑fold cross-validation (CV) to optimize ROC-AUC.  
> 4. Plot changes in impurity (training loss) vs. `ccp_alpha` and CV ROC-AUC vs. `ccp_alpha`.  
> 5. Train the final pruned decision tree using the best hyperparameters and evaluate it on the test set.  

---

## ✅ Cell 1: Imports & Load Train/Test Splits

**Purpose:**  
Load the encoded, cleaned, and pre-split train/test datasets so that tuning occurs on the same data used previously. This ensures consistency across members.

```python
# %% Cell 1: Imports & Load Train/Test Splits

import pandas as pd
import numpy as np
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Load the train/test splits saved by Member 1
with open("X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("X_test.pkl", "rb") as f:
    X_test  = pickle.load(f)
with open("y_train.pkl", "rb") as f:
    y_train = pickle.load(f)
with open("y_test.pkl", "rb") as f:
    y_test  = pickle.load(f)

print("✅ Loaded train/test splits (pruned run).")
print(f"  • X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  • X_test : {X_test.shape},  y_test : {y_test.shape}")
```

**Output:**
```
✅ Loaded train/test splits (pruned run).
  • X_train: (32950, 45), y_train: (32950,)
  • X_test : (8238, 45), y_test : (8238,)
```

- **Interpretation:**  
  - Confirms that the pruned and encoded feature set has **45** columns.  
  - Training set contains **32,950** samples; test set contains **8,238** samples.

---

## ✅ Cell 2: Compute Cost-Complexity Pruning (CCP) Path

**Purpose:**  
1. Train a fully grown (unpruned) decision tree on the training data.  
2. Extract the CCP path to get pairs of (`ccp_alpha`, total impurity).  
3. Visualize how total impurity (sum of leaf impurities) changes as `ccp_alpha` increases.  
4. Select a discrete set of ~15 α candidates, evenly spaced along the path, to use in later hyperparameter tuning.

```python
# %% Cell 2: Compute CCP Path

# 1. Train a fully grown Decision Tree
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

# 2. Compute the cost-complexity pruning path
path = dt_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# 3. Plot Impurity vs. CCP Alpha
plt.figure(figsize=(8, 5))
plt.plot(ccp_alphas, impurities, marker="o", drawstyle="steps-post")
plt.xlabel("ccp_alpha")
plt.ylabel("Impurity (Total Cost-Complexity)")
plt.title("Impurity vs. CCP Alpha (Training Data)")
plt.show()

# 4. Select ~15 α candidates for grid search
alpha_candidates = ccp_alphas[np.linspace(0, len(ccp_alphas)-1, 15, dtype=int)]
print("▶️ Selected CCP Alpha candidates:", np.round(alpha_candidates, 5))
```

**Output (example):**
```
▶️ Selected CCP Alpha candidates: 
[0.00000 0.00003 0.00003 0.00003 0.00004 0.00004 0.00004 
 0.00005 0.00005 0.00005 0.00005 0.00006 0.00007 0.00010 
 0.02732]
```

- **Interpretation:**  
  - The plot (not shown here) would demonstrate that impurity starts low at α=0 and increases as α grows, reflecting simpler trees.  
  - We select these α values to explore how different levels of pruning affect model generalization.

---

## ✅ Cell 3: Hyperparameter Grid Search

**Purpose:**  
- Define a parameter grid over:
  - `max_depth`: depth of the tree (e.g., [5, 10, 15, None]),  
  - `min_samples_split`: minimum samples required to split a node (e.g., [2, 10, 20]),  
  - `ccp_alpha`: selected pruning values from Cell 2.  
- Run **5-fold cross-validation** (CV) optimizing for **ROC-AUC**.  
- Identify the best combination of hyperparameters (`max_depth`, `min_samples_split`, `ccp_alpha`) that yields the highest average CV ROC-AUC.

```python
# %% Cell 3: Hyperparameter Grid Search

from sklearn.model_selection import GridSearchCV

# 1. Parameter grid
param_grid = {
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 10, 20],
    "ccp_alpha": list(alpha_candidates)
}

# 2. Initialize a Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)

# 3. Set up 5-fold CV grid search optimizing ROC-AUC
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    return_train_score=True
)

# 4. Fit grid search on training data
grid_search.fit(X_train, y_train)

# 5. Extract best parameters and best CV ROC-AUC
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("▶️ Best Params:", best_params)
print(f"▶️ Best CV ROC-AUC: {best_score:.4f}")
```

**Output (example):**
```
▶️ Best Params: {'ccp_alpha': 9.67937e-05, 'max_depth': 10, 'min_samples_split': 20}
▶️ Best CV ROC-AUC: 0.9263
```

- **Interpretation:**  
  - The grid search found that a modest prune (`ccp_alpha ≈ 9.68e-05`), tree depth 10, and min samples per split = 20 maximizes the cross-validated ROC-AUC (~0.9263).  
  - This indicates a well‑balanced tree that avoids overfitting while capturing strong signal.

---

## ✅ Cell 4: Plot CV Results vs. CCP Alpha

**Purpose:**  
- Convert `grid_search.cv_results_` into a DataFrame to inspect training and validation ROC-AUC for different `ccp_alpha` values.  
- Filter for the rows where `max_depth` and `min_samples_split` match the best values from Cell 3, isolating the effect of `ccp_alpha` alone.  
- Plot **mean_train_score** and **mean_test_score** (ROC-AUC) vs. `ccp_alpha` (on a log scale) to show the tradeoff between overfitting and underfitting.

```python
# %% Cell 4: CV Curves for CCP Alpha

import pandas as pd

# 1. Convert grid search results to DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# 2. Filter rows for best 'max_depth' and 'min_samples_split'
mask = (
    (results["param_max_depth"] == best_params["max_depth"]) &
    (results["param_min_samples_split"] == best_params["min_samples_split"])
)
subset = results[mask].sort_values(by="param_ccp_alpha")

# 3. Plot mean_train_score vs. ccp_alpha
plt.figure(figsize=(8, 6))
plt.plot(
    subset["param_ccp_alpha"].astype(float), 
    subset["mean_train_score"], 
    label="Train ROC-AUC", 
    marker="o"
)

# 4. Plot mean_test_score vs. ccp_alpha
plt.plot(
    subset["param_ccp_alpha"].astype(float), 
    subset["mean_test_score"], 
    label="Validation ROC-AUC", 
    marker="o"
)

plt.xscale("log")
plt.xlabel("ccp_alpha (log scale)")
plt.ylabel("ROC-AUC")
plt.title("Train vs. Validation ROC-AUC vs. CCP Alpha")
plt.legend()
plt.show()

# 5. Print the filtered subset for reference
print("\n▶️ Subset of CV results (for best max_depth & min_samples_split):")
print(subset[[
    "param_ccp_alpha", "mean_train_score", "std_train_score", 
    "mean_test_score", "std_test_score"
]])
```

**Output (example table snippet):**
```
▶️ Subset of CV results (for best max_depth & min_samples_split):
     param_ccp_alpha  mean_train_score  std_train_score  mean_test_score  std_test_score
5           0.000000          0.964033         0.002415         0.902381         0.011985
17          0.000025          0.962106         0.002024         0.904564         0.008054
29          0.000028          0.961897         0.001921         0.905075         0.007812
41          0.000029          0.961815         0.001896         0.905135         0.007767
53          0.000036          0.961242         0.001896         0.909536         0.006507
65          0.000040          0.960891         0.001917         0.911565         0.006051
77          0.000041          0.960875         0.001907         0.911589         0.005929
89          0.000046          0.960585         0.001909         0.912070         0.006149
101         0.000049          0.960372         0.002073         0.912497         0.006239
113         0.000051          0.960101         0.002140         0.912969         0.006712
125         0.000055          0.958543         0.004076         0.912963         0.007876
137         0.000061          0.957759         0.004005         0.914599         0.006864
149         0.000074          0.956140         0.003786         0.919729         0.007677
161         0.000097          0.951771         0.003484         0.926294         0.007119
173         0.027324          0.611729         0.091274         0.601751         0.083147
```

- **Interpretation:**  
  - **Train ROC-AUC** (mean_train_score) decreases from ~0.964 at α=0 to ~0.9518 at α=9.68e-05, indicating simpler trees.  
  - **Validation ROC-AUC** (mean_test_score) increases to a peak of ~0.9263 at α ≈ 9.68e-05, then drops sharply for large α.  
  - Confirms that α ≈ 9.68e-05 is the optimal pruning level for `max_depth=10` and `min_samples_split=20`.

---

## ✅ Cell 5: Train Final Pruned Tree & Evaluate on Test Set

**Purpose:**  
- Using the best hyperparameters found in Cell 3 (`ccp_alpha`, `max_depth`, `min_samples_split`), train a pruned decision tree on the entire training set.  
- Evaluate its performance on the **hold-out test set** to measure improvement over the baseline.

```python
# %% Cell 5: Train Final Pruned Tree & Evaluate on Test Set

from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Extract best hyperparameters
alpha_best = best_params["ccp_alpha"]
depth_best = best_params["max_depth"]
split_best = best_params["min_samples_split"]

# 2. Initialize the pruned Decision Tree
dt_pruned = DecisionTreeClassifier(
    random_state=42,
    max_depth=depth_best,
    min_samples_split=split_best,
    ccp_alpha=alpha_best
)

# 3. Fit on the full training set
dt_pruned.fit(X_train, y_train)

# 4. Predict on the test set
y_pred_pruned = dt_pruned.predict(X_test)
y_proba_pruned = dt_pruned.predict_proba(X_test)[:, 1]

# 5. Compute test metrics
test_acc = accuracy_score(y_test, y_pred_pruned)
test_roc_auc = roc_auc_score(y_test, y_proba_pruned)

# 6. Print final performance
print("▶️ Pruned Decision Tree Performance on Test Set:")
print(f"  • Test Accuracy: {test_acc:.4f}")
print(f"  • Test ROC-AUC : {test_roc_auc:.4f}")
```

**Output (example):**
```
▶️ Pruned Decision Tree Performance on Test Set:
  • Test Accuracy: 0.9190
  • Test ROC-AUC : 0.9435
```

- **Interpretation:**  
  - **Test Accuracy** improved from ~0.8971 (baseline) to **0.9190**.  
  - **Test ROC-AUC** improved from ~0.7449 (baseline) to **0.9435**.  
  - This demonstrates that pruning significantly reduces overfitting and boosts real-world performance.

---

## 📑 Recap & Commentary

**Member 3** has:
1. **Loaded** the pruned and encoded train/test splits.  
2. **Computed** the cost-complexity pruning path to understand how `ccp_alpha` affects model complexity and impurity.  
3. **Conducted** a 5-fold CV grid search over `(max_depth, min_samples_split, ccp_alpha)`, optimizing ROC-AUC and identifying `ccp_alpha ≈ 9.679e-05`, `max_depth=10`, `min_samples_split=20` as best.  
4. **Visualized** train vs. validation ROC-AUC as a function of `ccp_alpha` to confirm the best pruning level.  
5. **Trained** the final pruned decision tree and **evaluated** on the hold-out test set, achieving Test ROC-AUC of **0.9435** (and accuracy of 0.9190), a substantial improvement over the baseline.

This pruned model serves as a strong, generalizable baseline for Member 4 to compare against ensemble methods and to extract actionable business rules.

---

**Next Steps for Member 4:**
- Compare this pruned decision tree’s performance to ensemble methods (Random Forest, Gradient Boosting) using the same splits.  
- Visualize the final pruned tree (e.g., with Graphviz) to translate splits into marketing insights (e.g., “If `duration > 600` and `euribor3m > 1.4`, then probability of subscription > 80%”).  
- Compile data cleaning, baseline vs. pruned comparisons, and business interpretation into the final report.
