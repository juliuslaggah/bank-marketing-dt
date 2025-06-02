
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib  # for saving train/test splits

# Display all columns when showing DataFrames
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid", context="notebook")
```

- **pandas (pd)**: reading CSV, handling DataFrames  
- **numpy (np)**: numerical operations (e.g., creating NaNs, trigonometric functions)  
- **matplotlib.pyplot / seaborn**: plotting heatmaps and histograms  
- **LabelEncoder**: converting binary categories into 0/1  
- **train_test_split / joblib**: splitting data and saving splits for reuse  

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
3. **Categorical features contain the literal `'unknown'`** (not counted as `NaN`). For example:
   - `job` has categories like `“admin.”`, `“services”`, etc., and one category labeled `'unknown'`.  
   - `education` includes `'basic.4y'`, `'high.school'`, `'unknown'`, etc.  
   - `default` shows `'no'`, `'unknown'`, `'yes'`, etc.  
   - `housing`, `loan` likewise contain `'yes'`,`'no'`, and `'unknown'`.  
4. **Target column (`y`)** is object type (`'yes'/'no'`).  

Because pandas did not automatically flag any true “missing” (NaN) values, we now know that **“unknown”** must be treated explicitly as missing.

---

## ✅ Cell 4: Replace “unknown” with `np.nan` & Summarize Missing

### Purpose
- Convert all occurrences of the string `'unknown'` to actual `np.nan` so that missingness can be handled uniformly.  
- Re-count missing values to see how many entries fall into each column.

```python
# 1. Replace every “unknown” string with NumPy’s NaN
import numpy as np
df.replace("unknown", np.nan, inplace=True)

# 2. Check missing counts again
print("Missing values after replacing 'unknown':\n")
print(df.isnull().sum())

# 3. (Optional) Show % missing per column
print("\nPercentage of missing values per column:\n")
print((df.isnull().sum() / len(df) * 100).round(2))
```

#### Output Summary
| Column     | Missing Count | % Missing (approx.) |
|------------|---------------|----------------------|
| **job**       | 330           | 0.80%               |
| **marital**   | 80            | 0.19%               |
| **education** | 1 731         | 4.20%               |
| **default**   | 8 597         | 20.90%              |
| **housing**   | 990           | 2.40%               |
| **loan**      | 990           | 2.40%               |
| All others   | 0             | 0.00%               |

- Now we see actual missingness where “unknown” used to be.  
- In particular, **`default`** has ~21% missing—this is high.  

---

## ✅ Cell 5: Impute Missing Values with Mode (Avoiding Chained-Assignment Warning)

### Purpose
- Fill missing values (`np.nan`) in categorical columns with the **mode** (most frequent value).  
- Use a safe assignment pattern (`df[col] = df[col].fillna(...)`) rather than `inplace=True` on a chained call, to avoid the pandas `FutureWarning`.

```python
# Impute missing values in low‐ or moderate‐missing columns
cols_to_impute = ['education', 'job', 'marital', 'housing', 'loan']
for col in cols_to_impute:
    df[col] = df[col].fillna(df[col].mode()[0])

# For 'default' (20.9% missing), also fill with mode
df['default'] = df['default'].fillna(df['default'].mode()[0])

# Confirm that no missing values remain
print("✅ Missing values after imputation:\n")
print(df.isnull().sum())
```

#### Rationale
- **Low‐missing columns (<5%)**: `education`, `job`, `marital`, `housing`, `loan`.  
  Filling with mode ensures we don’t lose too many rows and preserves the most common category.  
- **High‐missing column (~21%)**: `default`.  
  We also fill with mode (“no” is likely most common), but note that other strategies could include:  
  - Dropping the column entirely (if not deemed important), or  
  - Creating a separate “Unknown” flag for marketing context.  

> After this step, all categorical variables have zero `NaN`s.

---

## ✅ Cell 6: Encode Categorical Variables

### Purpose
- Convert **binary** categorical columns to 0/1 using **LabelEncoder**.  
- Convert **multi-category** nominal columns to one-hot (dummy) variables.

#### 1. Identify Categorical Columns

```python
# List of columns that are still object dtype
cat_cols = df.select_dtypes(include='object').columns.tolist()
print("Categorical columns (object dtype):", cat_cols)
```

You should see (after imputation, but before encoding):
```
['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
 'month', 'day_of_week', 'poutcome', 'y']
```

#### 2. Label Encode Binary Columns

```python
from sklearn.preprocessing import LabelEncoder

# Columns with exactly two categories (yes/no)
binary_cols = ['default', 'housing', 'loan', 'y']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
    # Now df[col] contains 0/1 instead of “no”/“yes”

print("✅ Label-encoded binary columns:", binary_cols)
```

> After this:
> - `default`, `housing`, `loan`, and `y` have values in {0, 1}.

#### 3. One-Hot Encode Remaining Nominal Columns

```python
# Columns to one-hot encode (all remaining object-type columns except 'y')
one_hot_cols = ['job', 'marital', 'education', 'contact', 
                'month', 'day_of_week', 'poutcome']

# Perform one-hot encoding with drop_first=True to avoid redundancy
df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

# Display new shape
print("✅ One-hot encoding complete. New DataFrame shape:", df.shape)
```

- **`drop_first=True`**: we remove the first dummy to prevent perfect multicollinearity (though decision trees do not require strict avoidance of dummy traps, it keeps the feature set cleaner).  
- At this point, all columns in `df` are numeric (integers or floats).  

---

## ✅ Cell 7: Encode Target + Correlation Analysis + Feature Pruning

### Purpose
1. **Encode the target column** (`y`) from 0/1 (already done in Cell 6 with LabelEncoder).  
2. **Compute a correlation matrix** among numeric features.  
3. **Identify highly correlated feature pairs** (|corr| > 0.85) and drop redundancies.

#### 1. (Already done) Target encoding:
```python
# y was label-encoded in the previous step to 0/1
print("✅ Target variable 'y' is already encoded as 0/1.")
```

#### 2. Compute Correlation Matrix & Plot Heatmap
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Compute correlation matrix (numeric_only=True filters out any non-numeric, though all are numeric now)
corr_matrix = df.corr(numeric_only=True)

# Plot a large heatmap for visual inspection
plt.figure(figsize=(18, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of All Numeric Features", fontsize=16)
plt.show()
```

- The heatmap helps visually spot which features move together.  
- In particular, we expect to see very strong correlation among `emp.var.rate`, `euribor3m`, and `nr.employed`.

#### 3. Identify & Drop Highly Correlated Pairs
```python
# Only look at the upper triangle (k=1) to avoid duplicate pairs
mask_upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
upper_corr = corr_matrix.where(mask_upper)

# Filter pairs where |correlation| > 0.85
threshold = 0.85
high_corr_pairs = (
    upper_corr.stack()
    .reset_index()
    .rename(columns={'level_0': 'Feature1', 'level_1': 'Feature2', 0: 'Correlation'})
)
high_corr_pairs = high_corr_pairs[high_corr_pairs['Correlation'].abs() > threshold]

print("⚠️ Highly correlated feature pairs (|corr| > 0.85):\n")
print(high_corr_pairs.sort_values(by='Correlation', ascending=False))
```

Example printed output (rounded):

|   | Feature1      | Feature2     | Correlation |
|---|---------------|--------------|-------------|
| 0 | emp.var.rate  | euribor3m    | 0.972       |
| 1 | euribor3m     | nr.employed  | 0.945       |
| 2 | emp.var.rate  | nr.employed  | 0.907       |

These three features are nearly collinear. Keeping all three is redundant (and not ideal for some models), so we drop two of them.

```python
# Drop 'emp.var.rate' and 'nr.employed', keep only 'euribor3m'
df.drop(['emp.var.rate', 'nr.employed'], axis=1, inplace=True)

print("\n✅ Dropped 'emp.var.rate' and 'nr.employed' due to very high multicollinearity.")
print("New DataFrame shape:", df.shape)
```

> **Resulting DataFrame**: numeric, cleaned, with one of each strongly correlated trio (`euribor3m` remains).

---

## ✅ Cell 8: Train/Test Split & Save Splits

### Purpose
- Separate the cleaned, feature-engineered DataFrame into **X (features)** and **y (target)**  
- Perform an **80/20 stratified split** to preserve the same positive/negative ratio in train and test sets  
- **Save** the resulting splits as `.pkl` files so all team members can load them and replicate experiments exactly

```python
from sklearn.model_selection import train_test_split
import joblib

# 1. Separate X and y
X = df.drop('y', axis=1)
y = df['y']

print("Feature matrix shape (X):", X.shape)
print("Target vector shape (y):", y.shape)

# 2. Stratified 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n✅ Train/Test split complete:")
print(f"  • X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  • X_test : {X_test.shape}, y_test : {y_test.shape}")

# 3. Save splits to disk
joblib.dump(X_train, 'X_train.pkl')
joblib.dump(X_test,  'X_test.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test,  'y_test.pkl')

print("\n✅ Saved X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl")
```

- **`random_state=42`** ensures reproducibility: whenever anyone reloads these `.pkl` files, they will see the exact same train/test split.  
- **`stratify=y`** preserves the original ratio of “y = 1” vs. “y = 0” across train and test.  
- These `.pkl` files should be committed (if small) or shared with the team; they guarantee that everyone is training/evaluating on exactly the same data subsets.

---

## 📑 Recap & Commentary

By completing the cells above, **Member 1** has:

- **Imported** all necessary libraries  
- **Loaded** the raw “bank-full.csv” into a DataFrame  
- **Inspected** data structure and identified that missingness was encoded as the string **“unknown”**  
- **Replaced** “unknown” with `np.nan` and **imputed** missing values via the **mode** of each categorical column  
- **Label-encoded** binary columns (`default`, `housing`, `loan`, and the **target** `y`) to {0, 1}  
- **One-hot-encoded** all remaining nominal categorical columns (`job`, `marital`, `education`, `contact`, `month`, `day_of_week`, `poutcome`)  
- **Computed** a correlation matrix of numeric features, **identified** pairs with |corr| > 0.85 (`emp.var.rate` ↔ `euribor3m`, `euribor3m` ↔ `nr.employed`, `emp.var.rate` ↔ `nr.employed`), and **dropped** two of them (`emp.var.rate` and `nr.employed`)  
- **Split** the final cleaned/encoded dataset into a **stratified 80/20 train/test split**  
- **Saved** all four splits (`X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`) so that Members 2–4 can load them directly and begin modeling  

**Next Steps for Member 2**:  
- Load `X_train.pkl` and `y_train.pkl`  
- Train a baseline `DecisionTreeClassifier(random_state=42)`  
- Evaluate on `X_test.pkl` / `y_test.pkl`  
- Generate initial metrics (accuracy, precision, recall, F1, ROC-AUC) and a confusion matrix  

You can now include this entire section under a “Data Cleaning & Feature Engineering” heading in your project report or README. Feel free to adjust the wording slightly to match your group’s style.
