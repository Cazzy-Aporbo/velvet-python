import pandas as pd

# Load any dataset
df = pd.read_csv("your_data.csv")  # or pd.read_json("your_data.json")

# Quick look
print(df.head())       # first 5 rows
print(df.tail())       # last 5 rows
print(df.info())       # columns, types, null counts
print(df.describe())   # summary stats for numeric columns


# Missing values
print(df.isnull().sum())

# Duplicate rows
print(df.duplicated().sum())

# Unique value counts
for col in df.columns:
    print(col, df[col].nunique())
    
    
import matplotlib.pyplot as plt

numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    df[col].hist()
    plt.title(col)
    plt.show()
    
    
    
    
# Check ranges
for col in numeric_cols:
    print(f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean()}")

# Example: percentages add up to 100?
# Example: categorical counts make sense?

# Example: normalize numeric columns
normalized = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
print(normalized.head())

# Correlations for numeric data
print(df.corr())

# Group summaries for categorical relationships
print(df.groupby('category_col')['numeric_col'].mean())

# Log every step
print("Step 1: Loaded data")
print("Step 2: Checked missing values")
print("Step 3: Described numeric columns")
# Etc.




