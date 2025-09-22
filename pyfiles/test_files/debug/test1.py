import pandas as pd

# Load any dataset
df = pd.read_csv("your_data.csv")  # or pd.read_json("your_data.json")

# Quick look
print(df.head())       # first 5 rows
print(df.tail())       # last 5 rows
print(df.info())       # columns, types, null counts
print(df.describe())   # summary stats for numeric columns