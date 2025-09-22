import pandas as pd

# Load dataset
df = pd.read_csv("sample_data.csv")

# Quick look
print(df.head())          # first rows
print(df.info())          # types & missing values
print(df.describe())      # summary statistics

# Check for unusual values
print(df['age'].value_counts())
print(df['loan_approved'].value_counts())