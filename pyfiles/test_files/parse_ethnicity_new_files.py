# Script to parse ethnicity from NEW categorized data using chunking
import pandas as pd
import os
import re

input_file = "all_new_categorized_data_Apr30_2025.csv"
output_file = "all_new_categorized_data_with_ethnicity_Apr30_2025.csv"
summary_file = "new_files_combined_ethnicity_distribution_summary.txt"
chunk_size = 50000  # Process in chunks

# Potential ethnicity/race columns identified
ethnicity_columns = ["Ethnicity", "Race", "Patient Description"] # Add others if needed

# Regex patterns to extract ethnicity from combined fields like "Patient Description"
ethnicity_patterns = {
    "White": re.compile(r"\bwhite\b", re.IGNORECASE),
    "Black": re.compile(r"\bblack\b|african american\b", re.IGNORECASE),
    "Hispanic": re.compile(r"\bhispanic\b|latino\b|latina\b", re.IGNORECASE),
    "Asian": re.compile(r"\basian\b", re.IGNORECASE),
    "Native American/Alaska Native": re.compile(r"\bamerican indian\b|alaska native\b|native american\b", re.IGNORECASE),
    "Native Hawaiian/Pacific Islander": re.compile(r"\bnative hawaiian\b|pacific islander\b", re.IGNORECASE),
    "Multiracial": re.compile(r"\bmultiracial\b|two or more races\b", re.IGNORECASE),
}

def parse_ethnicity(row, cols):
    # Check explicit columns first
    for col in ["Ethnicity", "Race"]:
        if col in cols and pd.notna(row[col]):
            val_lower = str(row[col]).lower()
            for eth, pattern in ethnicity_patterns.items():
                if pattern.search(val_lower):
                    return eth
            # If no pattern match but column has value, return it cleaned
            if val_lower not in ["unknown", "not specified", "nan"]:
                 return str(row[col]).strip()

    # Check combined description columns
    if "Patient Description" in cols and pd.notna(row["Patient Description"]):
        desc_lower = str(row["Patient Description"]).lower()
        for eth, pattern in ethnicity_patterns.items():
            if pattern.search(desc_lower):
                return eth

    return "Unknown/Not Specified"

ethnicity_counts = pd.Series(dtype=int)
first_chunk = True

print(f"Processing {input_file} in chunks...")

try:
    # Ensure the output file is empty before starting
    if os.path.exists(output_file):
        os.remove(output_file)

    for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False, iterator=True):
        print(f"Processing chunk...")
        available_cols = [col for col in ethnicity_columns if col in chunk.columns]

        if not available_cols:
            chunk["Ethnicity_Combined"] = "Unknown/Not Specified"
        else:
            chunk["Ethnicity_Combined"] = chunk.apply(lambda row: parse_ethnicity(row, chunk.columns), axis=1)

        ethnicity_counts = ethnicity_counts.add(chunk["Ethnicity_Combined"].value_counts(), fill_value=0)

        if first_chunk:
            chunk.to_csv(output_file, index=False, mode="w") # Corrected mode="w"
            first_chunk = False
        else:
            chunk.to_csv(output_file, index=False, mode="a", header=False) # Corrected mode="a"

except UnicodeDecodeError:
    print("UTF-8 failed, trying latin1...")
    ethnicity_counts = pd.Series(dtype=int)
    first_chunk = True
    # Ensure the output file is empty before starting latin1 processing
    if os.path.exists(output_file):
        os.remove(output_file)

    for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False, iterator=True, encoding="latin1"):
        print(f"Processing chunk (latin1)...")
        available_cols = [col for col in ethnicity_columns if col in chunk.columns]
        if not available_cols:
            chunk["Ethnicity_Combined"] = "Unknown/Not Specified"
        else:
            chunk["Ethnicity_Combined"] = chunk.apply(lambda row: parse_ethnicity(row, chunk.columns), axis=1)
        ethnicity_counts = ethnicity_counts.add(chunk["Ethnicity_Combined"].value_counts(), fill_value=0)
        if first_chunk:
            chunk.to_csv(output_file, index=False, mode="w") # Corrected mode="w"
            first_chunk = False
        else:
            chunk.to_csv(output_file, index=False, mode="a", header=False) # Corrected mode="a"
except Exception as e:
    print(f"Error processing file {input_file}: {e}")

print(f"Finished processing. Combined data with ethnicity saved to {output_file}")

# Save summary
with open(summary_file, "w") as f:
    f.write("Ethnicity Distribution Summary (Combined Column) - New Files:\n")
    f.write(ethnicity_counts.to_string())

print(f"Ethnicity distribution summary saved to {summary_file}")

