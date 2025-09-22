"""
data_inspection_hipaa.py

Professional data inspection template:
- Loads any dataset (CSV/JSON)
- Checks general data quality (missing values, duplicates, types)
- Explores distributions
- Performs HIPAA-related checks for Protected Health Information (PHI)
"""

# -----------------------------
# 1. Imports
# -----------------------------
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# -----------------------------
# 2. Load dataset
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
        # Check if column name contains PHI keyword
        if any(k.lower() in col.lower() for k in HIPAA_KEYWORDS):
            phi_columns.append(col)
    return phi_columns

def check_phi_values(df):
    # Simple regex checks for PHI patterns
    suspicious_patterns = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'phone': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'date': r'\b\d{4}-\d{2}-\d{2}\b'  # YYYY-MM-DD
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
# 5. Visualization (Optional)
# -----------------------------
def visualize_numeric(df):
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[col].hist()
        plt.title(f"{col} distribution")
        plt.show()

# -----------------------------
# 6. Main Workflow
# -----------------------------
if __name__ == "__main__":
    # Example usage: replace with your dataset path
    file_path = "sample_data.csv"

    df = load_dataset(file_path)
    inspect_data(df)

    # HIPAA Checks
    phi_cols = check_phi_columns(df)
    phi_values = check_phi_values(df)

    print("\n=== Potential PHI Columns (based on column names) ===")
    if phi_cols:
        print(phi_cols)
    else:
        print("No obvious PHI columns found.")

    print("\n=== Potential PHI Values (based on regex patterns) ===")
    if phi_values:
        for col, type_found in phi_values.items():
            print(f"Column: {col}, Detected type: {type_found}")
    else:
        print("No obvious PHI patterns detected.")

    # Optional: visualize numeric distributions
    visualize_numeric(df)