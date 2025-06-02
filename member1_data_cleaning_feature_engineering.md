
# Member 1: Data Cleaning & Feature Engineering (Bank Marketing Dataset)

> **Objective:**  
> Prepare the raw Bank Marketing dataset for modeling. This includes:
> 1. Loading the data  
> 2. Inspecting structure and missing patterns  
> 3. Handling “unknown” values and imputing missing data  
> 4. Encoding categorical features (including the binary target)  
> 5. Identifying and removing highly correlated features  
> 6. Splitting into train/test sets and saving them for the team  

---

## ✅ Cell 1: Import Libraries

### Purpose
Set up all of the libraries and environment options needed for dataset manipulation, visualization, and preprocessing.

```python
# Data handling, numerical computing, and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning utilities
from sklearn.model_selection import train_test_split
import pickle

# Display all columns when showing DataFrames
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid", context="notebook")
```

- **pandas (pd)**: reading CSV, handling DataFrames  
- **numpy (np)**: numerical operations (e.g., creating NaNs)  
- **matplotlib.pyplot / seaborn**: plotting heatmaps and histograms  
- **train_test_split / pickle**: splitting data and saving splits for reuse  

> After running Cell 1, verify that no import errors appear and that library versions are up to date.

---

## ✅ Cell 2: Load Dataset

### Purpose
Load the Bank Marketing “full” CSV into a pandas DataFrame and perform a quick preview of the first few rows.

```python
# 1. Read the CSV file (semicolon-delimited)
df = pd.read_csv('bank-full.csv', sep=';')

# 2. Preview the first five rows
df.head()
```

#### Observations
- The DataFrame has **41,188 rows** and **21 columns** (20 predictors + 1 target).  
- Mixed data types:
  - **Numeric**: `age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`  
  - **Categorical (object)**: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`  
  - **Binary target (object)**: `y` (“yes” / “no”)  

Example of the first few rows:

|   | age  | job       | marital | education        | default | housing | loan | contact   | month | day_of_week | duration | campaign | pdays | previous | poutcome     | emp.var.rate | cons.price.idx | cons.conf.idx | euribor3m | nr.employed | y    |
|---|------|-----------|---------|------------------|---------|---------|------|-----------|-------|-------------|----------|----------|-------|----------|--------------|--------------|----------------|---------------|-----------|-------------|------|
| 0 | 56   | housemaid | married | basic.4y         | no      | no      | no   | telephone | may   | mon         | 261      | 1        | 999   | 0        | nonexistent  | 1.1          | 93.994         | −36.4         | 4.857     | 5191.0      | no   |
| 1 | 57   | services  | married | high.school      | unknown | no      | no   | telephone | may   | mon         | 149      | 1        | 999   | 0        | nonexistent  | 1.1          | 93.994         | −36.4         | 4.857     | 5191.0      | no   |
| 2 | 37   | services  | married | high.school      | no      | yes     | no   | telephone | may   | mon         | 226      | 1        | 999   | 0        | nonexistent  | 1.1          | 93.994         | −36.4         | 4.857     | 5191.0      | no   |
| 3 | 40   | admin.    | married | basic.6y         | no      | no      | no   | telephone | may   | mon         | 151      | 1        | 999   | 0        | nonexistent  | 1.1          | 93.994         | −36.4         | 4.857     | 5191.0      | no   |
| 4 | 56   | services  | married | high.school      | no      | no      | yes  | telephone | may   | mon         | 307      | 1        | 999   | 0        | nonexistent  | 1.1          | 93.994         | −36.4         | 4.857     | 5191.0      | no   |

---

## ✅ Cell 3: Inspect Dataset Structure & Missingness

### Purpose
- Examine the shape, data types, and non-null counts  
- View summary statistics for numerical and categorical columns  
- Identify any missing values immediately present in the raw CSV (before handling “unknown”)

```python
# 1. Print overall shape
print("Dataset shape:", df.shape)

# 2. Display DataFrame info: types and non-null counts
print("\n--- DataFrame Info ---")
df.info()

# 3. Display descriptive statistics (numeric + object)
print("\n--- Descriptive Statistics (including categorical) ---")
display(df.describe(include='all'))

# 4. Check for any actual NaNs
print("\n--- Missing Values Count (before replacing 'unknown') ---")
print(df.isnull().sum())

# 5. Check unique categories in some key columns to spot 'unknown'
print("\n--- Unique Values in 'job', 'education', 'poutcome' ---")
print("job:", df['job'].unique())
print("education:", df['education'].unique())
print("poutcome:", df['poutcome'].unique())
```

#### Key Findings from Cell 3
1. **Shape**: `(41188, 21)` → 41,188 samples, 21 columns.  
2. **No actual `NaN` values** (all columns show 41,188 non-null).  
3. **Categorical features contain the literal `'unknown'`** (not counted as `NaN`).  
4. **Target column (`y`)** is object type (`'yes'/'no'`).  

Because pandas did not automatically flag any true “missing” (NaN) values, we now know that **“unknown”** must be treated explicitly as missing.

---

## ✅ Cell 4: Consolidated Encoding & Cleaning

### Purpose
- Replace “unknown” with `np.nan`  
- Impute missing values by mode  
- One-hot encode all categorical features  
- Encode target `y` to 0/1  
- Drop highly correlated features to reduce redundancy  

```python
# Import numpy for missing values
import numpy as np

# 1. Replace any "unknown" with np.nan
df.replace("unknown", np.nan, inplace=True)

# 2. Impute missing values (mode) for categorical columns
for col in ['job', 'marital', 'education', 'default', 'housing', 'loan']:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. One-hot encode all categorical columns (except 'y')
cat_cols = ['job', 'marital', 'education', 'default', 
            'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 4. Encode target column 'y' to binary (0: no, 1: yes)
df_encoded['y'] = df['y'].map({'no': 0, 'yes': 1})

# 5. Confirm all columns are numeric
print("➡️ dtypes after encoding:", df_encoded.dtypes.value_counts())

# 6. Identify and drop highly correlated features (|corr| > 0.85)
corr_matrix = df_encoded.corr(numeric_only=True)
mask_upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
high_corr = (corr_matrix.where(mask_upper)
             .stack()
             .reset_index()
             .rename(columns={'level_0': 'Feature1', 
                              'level_1': 'Feature2', 
                              0: 'Correlation'}))
high_corr_pairs = high_corr.loc[high_corr['Correlation'].abs() > 0.85]

print("➡️ Highly correlated pairs (|corr| > 0.85):")
print(high_corr_pairs)

# Drop 'emp.var.rate' and 'nr.employed' if present
df_final = df_encoded.drop(columns=['emp.var.rate', 'nr.employed'], errors='ignore')

print("➡️ Final DataFrame shape:", df_final.shape)
print("➡️ dtypes after dropping:", df_final.dtypes.value_counts())
```

> **Check** that `df_final` has no `object` dtypes and that redundant features are dropped.

---

## ✅ Cell 5: Train/Test Split & Save Splits

### Purpose
- Separate `df_final` into features (**X**) and target (**y**)  
- Perform an **80/20 stratified split**  
- Save the splits to disk as `.pkl` files for team use  

```python
# Separate features and target
X = df_final.drop("y", axis=1)
y = df_final["y"]

print("➡️ X shape:", X.shape, "| y shape:", y.shape)

# Perform stratified split (80/20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("➡️ X_train:", X_train.shape, "| y_train:", y_train.shape)
print("➡️ X_test :", X_test.shape,  "| y_test :", y_test.shape)

# Save splits to disk
with open("X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open("X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open("y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("✅ Saved new pickle splits.")
```

- **`random_state=42`** ensures reproducibility.  
- **`stratify=y`** preserves class distribution in train/test.  
- The resulting `.pkl` files (`X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`) should be the only splits present.

---

## 📑 Recap & Commentary

By completing these steps, **Member 1** has:

- **Imported** necessary libraries  
- **Loaded** the raw dataset  
- **Replaced** “unknown” with `np.nan` and **imputed** missing categorical values via mode  
- **One-hot-encoded** all categorical features and **binary-encoded** the target `y`  
- **Dropped** highly correlated features (`emp.var.rate`, `nr.employed`)  
- **Split** the final numeric dataset into a stratified 80/20 train/test split  
- **Saved** the splits (`X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`) for use by the team  

**Next Steps for Member 2**:  
- Load the corrected `.pkl` files  
- Train and evaluate a default `DecisionTreeClassifier`  
- Continue with confusion matrix, feature importance, and tree visualization  

This document can be included under a **“Data Cleaning & Feature Engineering”** section in the final report.  
