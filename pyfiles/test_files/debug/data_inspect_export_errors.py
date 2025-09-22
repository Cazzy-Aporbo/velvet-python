import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os

# Example dataset path
file_path = "sample_data.csv"

# Load dataset with error handling
try:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type. Use CSV or JSON.")
    print("Dataset loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    df = None

if df is not None:
    # Inspect data
    try:
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
    except Exception as e:
        print(f"[ERROR] Data inspection failed: {e}")

    # HIPAA / PHI Checks
    HIPAA_KEYWORDS = ['name', 'first_name', 'last_name', 'dob', 'date_of_birth', 'ssn', 'social_security',
                      'address', 'phone', 'email', 'medical_record', 'mrn', 'patient_id', 'health_id']

    try:
        phi_columns = [col for col in df.columns if any(k.lower() in col.lower() for k in HIPAA_KEYWORDS)]
        print("\n=== Potential PHI Columns ===")
        print(phi_columns if phi_columns else "No obvious PHI columns found.")
    except Exception as e:
        print(f"[ERROR] PHI column check failed: {e}")

    try:
        for col in df.columns:
            if df[col].dtype == object:
                matches = df[col].astype(str).str.contains(r"\b\d{3}-\d{2}-\d{4}\b", na=False)
                if matches.any():
                    print(f"Potential SSN found in column: {col}")
    except Exception as e:
        print(f"[ERROR] PHI value check failed: {e}")

    # Visualize numeric columns
    try:
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            df[col].hist()
            plt.title(f"{col} distribution")
            plt.show()
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")

    # Export to multiple formats
    try:
        df.to_csv("output.csv", index=False)
        print("CSV exported successfully")
    except Exception as e:
        print(f"[ERROR] CSV export failed: {e}")

    try:
        df.to_excel("output.xlsx", index=False)
        print("Excel exported successfully")
    except Exception as e:
        print(f"[ERROR] Excel export failed: {e}")

    try:
        df.to_json("output.json", orient="records", lines=True)
        print("JSON exported successfully")
    except Exception as e:
        print(f"[ERROR] JSON export failed: {e}")

    try:
        df.to_parquet("output.parquet", index=False)
        print("Parquet exported successfully")
    except Exception as e:
        print(f"[ERROR] Parquet export failed: {e}")

    try:
        df.to_html("output.html", index=False)
        print("HTML exported successfully")
    except Exception as e:
        print(f"[ERROR] HTML export failed: {e}")