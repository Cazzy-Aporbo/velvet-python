import pandas as pd
import matplotlib.pyplot as plt

# Example: S1 Borneo data (replace with any season Google Sheet link)
url = "https://docs.google.com/spreadsheets/d/1EXAMPLE_SHEET_ID/export?format=csv"

# Load data
df = pd.read_csv(url)

# Quick look
print(df.head())

# Clean data: ensure numeric columns are floats
numeric_cols = ['ChW%', 'TC%', 'JV%', 'SurvSc']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing Survival Score
df = df.dropna(subset=['SurvSc'])

# Top 10 contestants by Survival Score
top10 = df.sort_values(by='SurvSc', ascending=False).head(10)
print(top10[['Contestant', 'Season', 'ChW%', 'TC%', 'JV%', 'SurvSc']])

# Plot: Survival Score vs Challenge Win %
plt.figure(figsize=(10,6))
plt.scatter(df['ChW%'], df['SurvSc'], color='purple', alpha=0.7)
for i, row in df.iterrows():
    if row['SurvSc'] >= df['SurvSc'].max() - 0.1:  # annotate top performers
        plt.annotate(row['Contestant'], (row['ChW%'], row['SurvSc']), fontsize=8)
plt.xlabel('Challenge Win %')
plt.ylabel('Survival Score')
plt.title('Survival Score vs Challenge Win %')
plt.grid(True)
plt.show()