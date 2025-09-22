import pandas as pd
import os

file_path = "sample_data.csv"

try:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully")
except FileNotFoundError as fnf:
    print(f"[ERROR] {fnf}")
except pd.errors.ParserError as pe:
    print(f"[ERROR] Parsing failed: {pe}")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")