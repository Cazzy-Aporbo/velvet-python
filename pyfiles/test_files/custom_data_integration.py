#!/usr/bin/env python3
"""
Women's Health Data Integration Script (Custom File List)

This version integrates a specific list of pre-processed files provided by the user.
"""

import os
import pandas as pd
from datetime import datetime

# Define the exact list of processed files to integrate
FILES_TO_INTEGRATE = [
    "/Users/cazandraaporbo/Desktop/FOXX_Health/coding_work/jup_files/delivery/documentation/womens_health_data_extraction/processed_data/cdc_combined_data.csv",
    "/Users/cazandraaporbo/Desktop/FOXX_Health/coding_work/jup_files/delivery/documentation/womens_health_data_extraction/processed_data/cdc_wonder_natality_data_processed.csv",
    "/Users/cazandraaporbo/Desktop/FOXX_Health/coding_work/jup_files/delivery/documentation/womens_health_data_extraction/processed_data/harvard_health_processed.csv",
    "/Users/cazandraaporbo/Desktop/FOXX_Health/coding_work/jup_files/delivery/documentation/womens_health_data_extraction/processed_data/nhs_inform_processed.csv",
    "/Users/cazandraaporbo/Desktop/FOXX_Health/coding_work/jup_files/delivery/documentation/womens_health_data_extraction/processed_data/pew_research_processed.csv",
    "/Users/cazandraaporbo/Desktop/FOXX_Health/coding_work/jup_files/delivery/documentation/womens_health_data_extraction/processed_data/nia_worksheets_processed.csv",
    "/Users/cazandraaporbo/Desktop/FOXX_Health/coding_work/jup_files/delivery/documentation/womens_health_data_extraction/processed_data/researchgate_processed.csv"
]

# Output directory
OUTPUT_DIR = "/Users/cazandraaporbo/Desktop/FOXX_Health/coding_work/jup_files/delivery/documentation/womens_health_data_extraction/integrated_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_combine(files):
    all_dfs = []
    for file_path in files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df["source_file"] = os.path.basename(file_path)
                all_dfs.append(df)
                print(f"✅ Loaded: {file_path} ({len(df)} records)")
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def save_combined_data(df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = os.path.join(OUTPUT_DIR, f"womens_health_combined_{timestamp}.csv")
    latest_path = os.path.join(OUTPUT_DIR, "womens_health_combined_latest.csv")
    df.to_csv(combined_path, index=False)
    df.to_csv(latest_path, index=False)
    print(f"📁 Saved integrated dataset to:
  - {combined_path}
  - {latest_path}")

def main():
    print("🔄 Starting integration of specified processed data files...")
    combined_df = load_and_combine(FILES_TO_INTEGRATE)
    if not combined_df.empty:
        print(f"✅ Total combined records: {len(combined_df)}")
        save_combined_data(combined_df)
    else:
        print("❌ No data combined. Please check file paths and content.")

if __name__ == "__main__":
    main()
