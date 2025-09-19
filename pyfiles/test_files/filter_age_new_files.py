# Script to filter NEW CSV data for age 16+
import pandas as pd
import os
import glob

input_dir = "/home/ubuntu/upload"
output_dir = "/home/ubuntu/new_files_filtered"
os.makedirs(output_dir, exist_ok=True)

# List of the 5 new files to process
new_files = [
    "_women_patient_symptom_dataset_1000_complete.csv",
    "female_accidental_deaths_2012-2023.csv",
    "llm_ready_medicine_data.csv",
    "worldbank_women_medical_conditions.csv",
    "healthcare_access_demographics_clean.csv"
]

# Potential age column names identified from previous analysis
age_columns = ["Age", "Age_Group", "age"]

summary_lines = []

def parse_age(age_val):
    if pd.isna(age_val):
        return None
    try:
        # Handle numeric ages directly
        if isinstance(age_val, (int, float)):
            return float(age_val)
        # Handle age ranges like "45-54"
        if isinstance(age_val, str) and "-" in age_val:
            parts = age_val.split("-")
            # Ensure the first part is numeric before converting
            if parts[0].strip().replace(".", "", 1).isdigit():
                 return float(parts[0].strip())
        # Handle strings like "55+" or just numbers as strings
        if isinstance(age_val, str):
             # Remove non-numeric characters like "," or "+" before checking
            age_str_cleaned = ".".join(part for part in age_val.split(".") if part.isdigit() or part == "") # Handle potential decimals
            age_str_cleaned = "".join(filter(lambda x: x.isdigit() or x == ".", age_str_cleaned))
            if age_str_cleaned and age_str_cleaned.replace(".", "", 1).isdigit():
                return float(age_str_cleaned)
    except ValueError:
        pass
    return None

for filename in new_files:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename.replace(".csv", "_filtered.csv"))
    summary_lines.append(f"Processing: {filename}")

    if not os.path.exists(input_path):
        summary_lines.append(f"  File not found: {input_path}")
        continue

    df = None # Initialize df to None
    try:
        # Try reading with default UTF-8 first
        df = pd.read_csv(input_path, low_memory=False)
    except UnicodeDecodeError:
        try:
            # Fallback to latin1 if UTF-8 fails
            df = pd.read_csv(input_path, encoding="latin1", low_memory=False)
        except Exception as e_latin1:
            summary_lines.append(f"  Error reading file {filename} with latin1: {e_latin1}")
            continue
    except Exception as e_utf8:
        summary_lines.append(f"  Error reading file {filename} with utf-8: {e_utf8}")
        continue

    if df is None: # Should not happen if reading succeeded, but good practice
        continue

    original_rows = len(df)
    summary_lines.append(f"  Original rows: {original_rows}")

    found_age_col = None
    for col in age_columns:
        if col in df.columns:
            found_age_col = col
            break

    if found_age_col:
        # Apply the parsing function
        df["parsed_age"] = df[found_age_col].apply(parse_age)
        # Filter based on parsed age
        df_filtered = df[df["parsed_age"] >= 16].copy()
        # Drop the temporary parsed_age column
        df_filtered.drop(columns=["parsed_age"], inplace=True)
        filtered_rows = len(df_filtered)
        summary_lines.append(f"  Found age column '{found_age_col}'. Filtered rows (>=16): {filtered_rows}")
        df_filtered.to_csv(output_path, index=False)
    else:
        # If no age column found, assume all data is relevant (or filter later if needed)
        summary_lines.append(f"  No recognized age column found. Copying all rows.")
        df.to_csv(output_path, index=False)
        filtered_rows = original_rows

    summary_lines.append(f"  Saved filtered data to: {output_path}")
    summary_lines.append("---")

# Save summary
summary_file = "/home/ubuntu/new_files_age_filter_summary.txt"
with open(summary_file, "w") as f:
    f.write("\n".join(map(str, summary_lines))) # Ensure all elements are strings

print(f"Age filtering summary saved to {summary_file}")

