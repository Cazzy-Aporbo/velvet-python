# Script to transform NEW consolidated CSV data to JSON Lines format
import pandas as pd
import json
import os
import numpy as np

input_file = "/home/ubuntu/new_files_categorized/all_new_categorized_data_with_ethnicity_Apr30_2025.csv"
output_file = "/home/ubuntu/new_files_jsonl/new_files_data_Apr30_2025.jsonl"
chunk_size = 10000 # Process in chunks

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Define the JSON schema structure (based on json_schema_Apr30_2025.md)
def create_json_record(row):
    # Helper to safely get values
    def safe_get(key, default="Unknown"):
        val = row.get(key, default)
        return default if pd.isna(val) else val

    record = {
        "record_id": str(row.name), # Use index as a simple ID for now
        "patient_profile": {
            "age": safe_get("Age", safe_get("age", safe_get("Age_Group"))),
            "ethnicity": safe_get("Ethnicity_Combined", "Unknown/Not Specified"),
            "location": safe_get("Residence State", safe_get("LocationDesc", safe_get("Country Name"))), # Optional
            "income": safe_get("Income"), # Optional
            "education": safe_get("Education"),
            "insurance": safe_get("Insurance")
        },
        "health_context": {
            "symptoms": [],
            "conditions": [],
            "medications_vitamins": [],
            "body_region_category": safe_get("Body_Region_Category", "unknown"),
            "symptom_type_category": safe_get("Symptom_Type_Category", "unknown")
        },
        "interaction_context": {
            "dismissal_experience": safe_get("Dismissed Symptoms", None),
            "doctor_interaction_notes": safe_get("Patient Description", None),
            "previous_questions": safe_get("Diagnostic Questions", None)
        },
        "text_input": "", # To be constructed
        "target_question": None # Placeholder for augmentation
    }

    # Populate symptoms
    symptom_desc = ""
    symptom_type = record["health_context"]["symptom_type_category"]
    body_region = record["health_context"]["body_region_category"]
    if symptom_type != "unknown":
        symptom_desc += f"Experiencing {symptom_type.replace('_', ' ')}"
        if body_region != "unknown":
            region_formatted = body_region.replace("_", " ")
            symptom_desc += f" in the {region_formatted} region."
        else:
            symptom_desc += "."
        record["health_context"]["symptoms"].append(symptom_desc)

    injury_desc = safe_get("Description of Injury", None)
    if injury_desc:
         record["health_context"]["symptoms"].append(f"Injury: {injury_desc}")

    # Populate conditions
    condition = safe_get("condition", None)
    if condition:
        record["health_context"]["conditions"].append(condition)
    cause_of_death = safe_get("Cause of Death", None)
    if cause_of_death:
         record["health_context"]["conditions"].append(f"Cause of Death (Context): {cause_of_death}")
    other_conditions = safe_get("Other Significant Conditions", None)
    if other_conditions:
         record["health_context"]["conditions"].append(f"Other Significant Conditions: {other_conditions}")

    # Populate medications
    med_name = safe_get("name", None)
    if med_name:
        med_info = {"name": med_name, "uses": safe_get("uses", None), "side_effects": safe_get("side_effects", None)}
        record["health_context"]["medications_vitamins"].append(med_info)

    # Construct text_input
    text_input_parts = []
    profile = record["patient_profile"]
    health = record["health_context"]
    interaction = record["interaction_context"]

    age_val = profile["age"]
    age_str = f"{age_val}-year-old" if age_val != "Unknown" else "Patient"
    eth_val = profile["ethnicity"]
    eth_str = f" {eth_val}" if eth_val not in ["Unknown/Not Specified", "Unknown"] else ""
    text_input_parts.append(f"{age_str}{eth_str} presents with the following:")

    if health["symptoms"]:
        text_input_parts.append("Symptoms: " + "; ".join(health["symptoms"]))
    if health["conditions"]:
        text_input_parts.append("Conditions: " + "; ".join(health["conditions"]))
    if health["medications_vitamins"]:
        med_strs = []
        for med in health["medications_vitamins"]:
            uses_str = med["uses"] if med.get("uses") else "N/A"
            se_str = med["side_effects"] if med.get("side_effects") else "N/A"
            med_strs.append(f"{med['name']} (Uses: {uses_str}, Side Effects: {se_str})")
        text_input_parts.append("Medications/Vitamins: " + "; ".join(med_strs))

    dismissal = interaction["dismissal_experience"]
    if dismissal:
        text_input_parts.append(f"Previously dismissed symptoms: {dismissal}")
    interaction_notes = interaction["doctor_interaction_notes"]
    if interaction_notes:
        text_input_parts.append(f"Interaction notes: {interaction_notes}")
    prev_q = interaction["previous_questions"]
    if prev_q:
         text_input_parts.append(f"Previous diagnostic questions asked: {prev_q}")

    record["text_input"] = " ".join(text_input_parts)

    # Clean up potential NaN/NaT values before JSON serialization
    def clean_nans(obj):
        if isinstance(obj, dict):
            return {k: clean_nans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nans(elem) for elem in obj]
        elif pd.isna(obj):
            return None
        return obj

    return clean_nans(record)

print(f"Starting transformation of {input_file} to {output_file}")

# Clear output file if it exists
if os.path.exists(output_file):
    os.remove(output_file)

processed_count = 0
try:
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False, iterator=True):
        print(f"Processing chunk {processed_count // chunk_size + 1}...")
        json_records = chunk.apply(create_json_record, axis=1)
        with open(output_file, "a") as f:
            for record in json_records:
                try:
                    json.dump(record, f)
                    f.write("\n")
                except TypeError as e:
                    print(f"Serialization Error for record ID: {record.get('record_id', 'N/A')}. Error: {e}. Skipping record.")
                    pass
        processed_count += len(chunk)

except UnicodeDecodeError:
    print("UTF-8 failed, trying latin1...")
    if os.path.exists(output_file):
        os.remove(output_file)
    processed_count = 0
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False, iterator=True, encoding="latin1"):
        print(f"Processing chunk {processed_count // chunk_size + 1} (latin1)...")
        json_records = chunk.apply(create_json_record, axis=1)
        with open(output_file, "a") as f:
            for record in json_records:
                 try:
                    json.dump(record, f)
                    f.write("\n")
                 except TypeError as e:
                    print(f"Serialization Error for record ID: {record.get('record_id', 'N/A')}. Error: {e}. Skipping record.")
                    pass
        processed_count += len(chunk)
except Exception as e:
    print(f"An error occurred during transformation: {e}")

print(f"Transformation complete. {processed_count} records processed.")
print(f"Output saved to {output_file}")

