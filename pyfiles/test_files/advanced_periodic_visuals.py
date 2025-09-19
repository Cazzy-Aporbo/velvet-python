import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Generate a pastel rainbow color palette
pastel_rainbow = sns.color_palette("rainbow", 10)  # Generate discrete pastel colors
sns.set_theme(style="whitegrid", palette=pastel_rainbow)

# Load periodic table data
elements_data = [
    [1, "H", "Hydrogen", 1.008, 0.0899, 2.20, 20.28, 14.01],
    [2, "He", "Helium", 4.0026, 0.1786, None, 4.22, None],
    [3, "Li", "Lithium", 6.94, 0.534, 0.98, 1615, 453.69],
    [4, "Be", "Beryllium", 9.0122, 1.85, 1.57, 2742, 1560],
    [5, "B", "Boron", 10.81, 2.34, 2.04, 4200, 2349],
    [6, "C", "Carbon", 12.011, 2.267, 2.55, 4300, 3823],
    [7, "N", "Nitrogen", 14.007, 1.251, 3.04, 77.36, 63.15],
    [8, "O", "Oxygen", 15.999, 1.429, 3.44, 90.20, 54.36],
    [9, "F", "Fluorine", 18.998, 1.696, 3.98, 85.03, 53.48],
    [10, "Ne", "Neon", 20.180, 0.9002, None, 27.07, None]
]

# Create DataFrame
df = pd.DataFrame(elements_data, columns=[
    "atomic_number", "symbol", "name", "atomic_weight", "density", "electronegativity", "boiling_point", "melting_point"
])

# Handle missing values using KNN Imputation
imputer = KNNImputer(n_neighbors=3)
df.iloc[:, 3:] = imputer.fit_transform(df.iloc[:, 3:])

# Standardize numerical columns
scaler = StandardScaler()
df.iloc[:, 3:] = scaler.fit_transform(df.iloc[:, 3:])

# Save to CSV
file_path = "/Users/cazandraaporbo/Desktop/Summer/data_viz/reliable_periodic_table.csv"
os.makedirs(os.path.dirname(file_path), exist_ok=True)
df.to_csv(file_path, index=False)

print("\nProcessed Periodic Table Data:")
print(df.head())

# Generate visualizations
plt.figure(figsize=(12, 6))
sns.barplot(x="atomic_number", y="atomic_weight", data=df, palette="rainbow")
plt.title("Atomic Weights Across the Periodic Table (First 50 Elements)")
plt.xlabel("Atomic Number")
plt.ylabel("Standardized Atomic Weight")
plt.xticks(rotation=90)
plt.show()

# 3D Scatter Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["atomic_number"], df["electronegativity"], df["atomic_weight"], c=df["atomic_number"], cmap="rainbow", s=50)
ax.set_xlabel("Atomic Number")
ax.set_ylabel("Electronegativity")
ax.set_zlabel("Atomic Weight")
ax.set_title("3D Periodic Table Visualization (First 50 Elements)")
plt.show()
