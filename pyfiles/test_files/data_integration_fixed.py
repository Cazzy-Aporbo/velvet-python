
#!/usr/bin/env python3

import os
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
INTEGRATED_DATA_DIR = os.path.join(BASE_DIR, "integrated_data")

os.makedirs(INTEGRATED_DATA_DIR, exist_ok=True)

COLUMN_MAPPING = {
    "source": "source",
    "category": "category",
    "topic": "topic",
    "description": "description",
    "content": "content",
    "authors": "authors",
    "journal": "journal",
    "doi": "doi",
    "citations": "citations",
    "keywords": "keywords",
    "file_path": "file_path",
    "demographics": "demographics",
    "statistics": "statistics",
    "recommendations": "recommendations",
    "references": "references",
    "date": "date",
    "extraction_date": "extraction_date",
    "url": "url"
}

def load_and_standardize_csv(file_path, source_name):
    try:
        df = pd.read_csv(file_path)
        df["source"] = source_name  # overwrite or add source label
        df = df.rename(columns={col: COLUMN_MAPPING.get(col, col) for col in df.columns})
        return df
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return pd.DataFrame()

def integrate_data():
    all_data = []
    sources = [
        "CDC Open Data",
        "NIA Talking with Your Doctor Worksheets",
        "NHS Inform",
        "Pew Research Reports",
        "ResearchGate Medical Studies",
        "Harvard Health Blog",
        "Harvard Health Blog V2"
    ]

    for source in sources:
        filename = f"{source.lower().replace(' ', '_')}_processed.csv"
        file_path = os.path.join(PROCESSED_DATA_DIR, filename)
        if os.path.exists(file_path):
            print(f"✅ Loading: {filename}")
            df = load_and_standardize_csv(file_path, source)
            if not df.empty:
                all_data.append(df)
        else:
            print(f"⚠️ Skipped (not found): {filename}")

    if all_data:
        integrated_df = pd.concat(all_data, ignore_index=True).drop_duplicates()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        integrated_file = os.path.join(INTEGRATED_DATA_DIR, f"integrated_data_{timestamp}.csv")
        latest_file = os.path.join(INTEGRATED_DATA_DIR, "integrated_data_latest.csv")
        integrated_df.to_csv(integrated_file, index=False)
        integrated_df.to_csv(latest_file, index=False)
        print(f"✅ Integrated {len(integrated_df)} total records")
        print(f"📁 Saved to:
  - {integrated_file}
  - {latest_file}")
    else:
        print("❌ No data found to integrate.")

if __name__ == "__main__":
    integrate_data()
