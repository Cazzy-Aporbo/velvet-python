import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load dataset (replace with your dataset or URL)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url)

# Filter for luxury brands (example: BMW, Audi, Mercedes)
luxury_brands = ["audi", "bmw", "mercedes-benz", "jaguar", "lexus", "porsche"]
df['manufacturer'] = df['name'].str.split().str[0].str.lower()
df_luxury = df[df['manufacturer'].isin(luxury_brands)].copy()

# Clean data: fill missing values
df_luxury['horsepower'] = pd.to_numeric(df_luxury['horsepower'], errors='coerce')
df_luxury['horsepower'].fillna(df_luxury['horsepower'].median(), inplace=True)
df_luxury['weight'] = pd.to_numeric(df_luxury['weight'], errors='coerce')
df_luxury['weight'].fillna(df_luxury['weight'].median(), inplace=True)
df_luxury['mpg'].fillna(df_luxury['mpg'].median(), inplace=True)

# Analysis
top_brands = df_luxury['manufacturer'].value_counts()
avg_mpg_by_brand = df_luxury.groupby('manufacturer')['mpg'].mean()
avg_weight_by_brand = df_luxury.groupby('manufacturer')['weight'].mean()

# Visualizations
plt.figure(figsize=(10,5))
top_brands.plot.barh()
plt.title("Top Luxury Brands by Count in Dataset")
plt.xlabel("Number of Cars")
plt.tight_layout()
plt.savefig("top_luxury_brands.png")
plt.close()

plt.figure(figsize=(10,5))
avg_mpg_by_brand.plot.barh()
plt.title("Average MPG by Luxury Brand")
plt.xlabel("MPG")
plt.tight_layout()
plt.savefig("avg_mpg_by_brand.png")
plt.close()

plt.figure(figsize=(10,5))
avg_weight_by_brand.plot.barh()
plt.title("Average Weight by Luxury Brand")
plt.xlabel("Weight")
plt.tight_layout()
plt.savefig("avg_weight_by_brand.png")
plt.close()

# Generate HTML report
html_content = f"""
<html>
<head><title>Luxury Car Data Report</title></head>
<body>
<h1>Luxury Car Data Report</h1>
<p>Report generated: {datetime.now()}</p>
<h2>Top Luxury Brands</h2>
<img src='top_luxury_brands.png' width='600'><br>
<h2>Average MPG by Brand</h2>
<img src='avg_mpg_by_brand.png' width='600'><br>
<h2>Average Weight by Brand</h2>
<img src='avg_weight_by_brand.png' width='600'><br>
<h2>Sample Data</h2>
{df_luxury.head(20).to_html()}
</body>
</html>
"""

with open("luxury_car_report.html", "w") as f:
    f.write(html_content)

print("Report generated: luxury_car_report.html")