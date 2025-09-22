"""
data_inspect_export.py

Professional data inspection + HIPAA check + multi-format export template
"""

# -----------------------------
# 1. Imports
# -----------------------------
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# -----------------------------
# 2. Load Dataset
# -----------------------------
def load_dataset(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type. Use CSV or JSON.")
    return df

# -----------------------------
# 3. General Data Inspection
# -----------------------------
def inspect_data(df):
    print("\n=== Data Info ===")
    print(df.info())
    print("\n=== Head of Data ===")
    print(df.head())
    print("\n=== Summary Statistics ===")
    print(df.describe(include='all'))
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    print("\n=== Duplicate Rows ===")
    print(df.duplicated().sum())

# -----------------------------
# 4. HIPAA / PHI Checks
# -----------------------------
HIPAA_KEYWORDS = [
    'name', 'first_name', 'last_name', 'dob', 'date_of_birth',
    'ssn', 'social_security', 'address', 'phone', 'email',
    'medical_record', 'mrn', 'patient_id', 'health_id'
]

def check_phi_columns(df):
    phi_columns = []
    for col in df.columns:
        if any(k.lower() in col.lower() for k in HIPAA_KEYWORDS):
            phi_columns.append(col)
    return phi_columns

def check_phi_values(df):
    suspicious_patterns = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'phone': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'date': r'\b\d{4}-\d{2}-\d{2}\b'
    }
    findings = {}
    for col in df.columns:
        col_values = df[col].astype(str)
        for key, pattern in suspicious_patterns.items():
            matches = col_values.str.contains(pattern)
            if matches.any():
                findings[col] = key
    return findings

# -----------------------------
# 5. Visualization
# -----------------------------
def visualize_numeric(df):
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[col].hist()
        plt.title(f"{col} distribution")
        plt.show()

# -----------------------------
# 6. Export Functions
# -----------------------------
def export_csv(df, filename="output.csv", index=False):
    df.to_csv(filename, index=index)
    print(f"CSV exported: {filename}")

def export_excel(df, filename="output.xlsx", index=False):
    df.to_excel(filename, index=index)
    print(f"Excel exported: {filename}")

def export_json(df, filename="output.json", orient="records"):
    df.to_json(filename, orient=orient, lines=True)
    print(f"JSON exported: {filename}")

def export_parquet(df, filename="output.parquet", index=False):
    df.to_parquet(filename, index=index)
    print(f"Parquet exported: {filename}")

def export_html(df, filename="output.html"):
    df.to_html(filename, index=False)
    print(f"HTML exported: {filename}")

# -----------------------------
# 7. Main Workflow
# -----------------------------
if __name__ == "__main__":
    # Replace with your dataset path
    file_path = "sample_data.csv"

    df = load_dataset(file_path)
    inspect_data(df)

    # HIPAA Checks
    phi_cols = check_phi_columns(df)
    phi_values = check_phi_values(df)

    print("\n=== Potential PHI Columns ===")
    print(phi_cols if phi_cols else "No obvious PHI columns found.")

    print("\n=== Potential PHI Values ===")
    if phi_values:
        for col, type_found in phi_values.items():
            print(f"Column: {col}, Detected type: {type_found}")
    else:
        print("No obvious PHI patterns detected.")

    # Optional: visualize numeric distributions
    visualize_numeric(df)

    # Export to all formats
    export_csv(df, "output.csv")
    export_excel(df, "output.xlsx")
    export_json(df, "output.json")
    export_parquet(df, "output.parquet")
    export_html(df, "output.html")