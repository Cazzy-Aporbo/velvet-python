# Script to create updated data source documentation
import pandas as pd

# List all unique CSV files provided by the user
files = [
    # Initial batch
    "improved_doctor_questions.csv",
    "upmc_checklist_llm_formatted.csv",
    "postnatal_data_no_date.csv",
    "worldbank_women_disease_symptoms.csv",
    "Health_Female_Cleaned.csv",
    "healthcare_access_disparities_dataset_1000.csv",
    "Female Pharma-safe Index Dataset.csv",
    "enhanced_healthcare_dataset.csv",
    "doctor_patient_llm_augmented.csv",
    "diagnostic_algorithm_performance_dataset.csv",
    "comprehensive_women_patient_symptom_dataset_balanced.csv",
    "comprehensive_dismissal_experiences_with_adolescents.csv",
    "cleaned_medical_conditions_female.csv",
    "cleaned_health_indicators.csv",
    "cleaned_drugs_structured.csv",
    "cdc_female_chronic_disease_indicators.csv",
    "anxiety_depression_female_only.csv",
    # Second batch
    "fully_completed_supplements_dataset.csv",
    "merged_llm_dataset.csv",
    "medicine_dataset.csv",
    "medquad.csv",
    # Third batch (excluding duplicates)
    "_women_patient_symptom_dataset_1000_complete.csv",
    "female_accidental_deaths_2012-2023.csv",
    "llm_ready_medicine_data.csv",
    "worldbank_women_medical_conditions.csv",
    "healthcare_access_demographics_clean.csv"
]

# Create metadata for each file (educated guesses based on names)
source_data = []
for filename in files:
    source = "User Upload - Unknown"
    license_use = "Unknown - Check Required"
    reliability = "Unknown"

    if "worldbank" in filename.lower():
        source = "User Upload - World Bank"
        license_use = "Public Domain / CC-BY (Check Specifics)"
        reliability = "Official Indicator/Survey Data"
    elif "cdc" in filename.lower():
        source = "User Upload - CDC"
        license_use = "Public Domain (US Gov)"
        reliability = "Official Health Indicator"
    elif "medquad" in filename.lower():
        source = "User Upload - MedQuAD Dataset (Research)"
        license_use = "Research Use (Check Specifics)"
        reliability = "Medical Q&A (Mixed)"
    elif "pharma" in filename.lower() or "drug" in filename.lower() or "medicine" in filename.lower() or "supplement" in filename.lower():
        source = "User Upload - Likely Drug/Supplement Database"
        license_use = "Unknown - Check Required"
        reliability = "Medication/Supplement Info"
    elif "augmented" in filename.lower() or "llm" in filename.lower() or "formatted" in filename.lower():
        source = "User Upload - Likely Synthesized/Augmented"
        license_use = "Assumed Permissive (User Generated)"
        reliability = "Synthesized/Processed Data"
    elif "symptom" in filename.lower() or "condition" in filename.lower() or "health" in filename.lower() or "patient" in filename.lower() or "dismissal" in filename.lower() or "postnatal" in filename.lower() or "diagnostic" in filename.lower() or "disparities" in filename.lower() or "demographics" in filename.lower():
        source = "User Upload - Likely Research/Survey Data"
        license_use = "Unknown - Check Required"
        reliability = "Research/Survey Data"
    elif "death" in filename.lower():
        source = "User Upload - Likely Vital Statistics/Mortality Data"
        license_use = "Public Record (Check Specifics)"
        reliability = "Mortality Data"

    source_data.append({
        "Filename": filename,
        "Assumed_Source": source,
        "Assumed_License_Commercial_Use": license_use,
        "Medical_Reliability_Note": reliability
    })

# Create DataFrame and save to CSV
df_sources = pd.DataFrame(source_data)
df_sources.to_csv("/home/ubuntu/data_sources_updated_Apr30_2025.csv", index=False)

print("Updated data source documentation created: /home/ubuntu/data_sources_updated_Apr30_2025.csv")

