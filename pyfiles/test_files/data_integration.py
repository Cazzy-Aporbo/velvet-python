#!/usr/bin/env python3
"""Women's Health Data Integration Script
...
(remainder of script trimmed for brevity, but will include all the pasted content)
"""

#!/usr/bin/env python3
"""
Women's Health Data Integration Script

This script integrates data extracted from multiple health information sources:
1. CDC Open Data
2. NIA Talking with Your Doctor Worksheets
3. NHS Inform
4. Pew Research Reports
5. ResearchGate Medical Studies
6. Harvard Health Blog

It combines all sources into a single comprehensive CSV file for analysis,
ensuring consistent data structure and proper handling of different formats.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import re

# Create directories for data storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
INTEGRATED_DATA_DIR = os.path.join(BASE_DIR, "integrated_data")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(INTEGRATED_DATA_DIR, exist_ok=True)

# Define standard column names for the integrated dataset
STANDARD_COLUMNS = [
    "source",              # Original data source
    "category",            # Health category
    "topic",               # Specific health topic or condition
    "description",         # Brief description
    "content",             # Main content
    "symptoms",            # Related symptoms
    "treatment",           # Treatment information
    "recommendations",     # Advice or recommended questions
    "demographics",        # Target demographic group
    "statistics",          # Statistical data
    "references",          # Original source references
    "date",                # Publication or data collection date
    "extraction_date"      # When the data was extracted
]

def log_message(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_processed_data(source_name):
    """
    Load processed data for a specific source.
    
    Args:
        source_name: Name of the source
    
    Returns:
        DataFrame with the processed data
    """
    file_pattern = os.path.join(PROCESSED_DATA_DIR, f"{source_name.lower().replace(' ', '_')}_processed.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        # Try alternative pattern
        file_pattern = os.path.join(PROCESSED_DATA_DIR, f"{source_name.lower().replace(' ', '_')}*.csv")
        files = glob.glob(file_pattern)
    
    if not files:
        log_message(f"No processed data found for {source_name}")
        return pd.DataFrame()
    
    # If multiple files, load and concatenate them
    dfs = []
    for file in files:
        try:
            log_message(f"Loading {file}")
            df = pd.read_csv(file, encoding='utf-8')
            dfs.append(df)
        except Exception as e:
            log_message(f"Error loading {file}: {str(e)}")
    
    if not dfs:
        return pd.DataFrame()
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    log_message(f"Loaded {len(combined_df)} records from {source_name}")
    
    return combined_df

def standardize_dataframe(df, source_name):  # sourcery skip: low-code-quality
    """
    Standardize a DataFrame to match the required structure.
    
    Args:
        df: DataFrame to standardize
        source_name: Name of the source
    
    Returns:
        Standardized DataFrame
    """
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    # Create a new DataFrame with standard columns
    standardized_df = pd.DataFrame(columns=STANDARD_COLUMNS)

    # Map existing columns to standard columns
    column_mapping = {
        # Common mappings
        "source": "source",
        "category": "category",
        "topic": "topic",
        "description": "description",
        "content": "content",
        "symptoms": "symptoms",
        "treatment": "treatment",
        "recommendations": "recommendations",
        "demographics": "demographics",
        "statistics": "statistics",
        "references": "references",
        "date": "date",
        "extraction_date": "extraction_date",

        # Source-specific mappings
        "title": "topic",
        "abstract": "description",
        "url": "references",
        "publication_date": "date",
        "authors": "references",
        "journal": "references",
        "key_points": "content",
        "when_to_seek_help": "recommendations",
        "questions": "recommendations",
        "file_path": None,  # Ignore this column
        "dataset": None,    # Ignore this column
        "causes": "content",
        "is_womens_health": None  # Ignore this column
    }

    # Copy data with column mapping
    for std_col in STANDARD_COLUMNS:
        # Find matching columns in the source DataFrame
        matching_cols = [col for col, mapped_col in column_mapping.items() 
                        if mapped_col == std_col and col in df.columns]

        standardized_df[std_col] = df[matching_cols[0]] if matching_cols else ""
    # Ensure source column is correct
    standardized_df["source"] = source_name

    # Handle special cases for each source
    if source_name == "CDC Open Data":
        # Combine location and statistics
        if "location" in df.columns and "statistics" in df.columns:
            standardized_df["statistics"] = df.apply(
                lambda row: f"Location: {row.get('location', '')}, Value: {row.get('statistics', '')}", 
                axis=1
            )

    elif source_name == "NIA Talking with Your Doctor Worksheets":
        # Convert questions list to recommendations
        if "questions" in df.columns:
            standardized_df["recommendations"] = df["questions"].apply(
                lambda x: x if isinstance(x, str) else ""
            )

    elif source_name == "NHS Inform":
        # Combine symptoms and treatment
        if "symptoms" in df.columns and "treatment" in df.columns:
            standardized_df["content"] = df.apply(
                lambda row: f"Symptoms: {row.get('symptoms', '')}\n\nTreatment: {row.get('treatment', '')}", 
                axis=1
            )

    elif source_name == "Pew Research Reports":
        # Use statistics as content if available
        if "statistics" in df.columns:
            standardized_df["content"] = df["statistics"].apply(
                lambda x: x if isinstance(x, str) else ""
            )

    elif source_name == "ResearchGate Medical Studies":
        # Combine author and journal information
        if "authors" in df.columns and "journal" in df.columns:
            standardized_df["references"] = df.apply(
                lambda row: f"Authors: {row.get('authors', '')}, Journal: {row.get('journal', '')}, URL: {row.get('references', '')}", 
                axis=1
            )

    elif source_name == "Harvard Health Blog":
        # Add author to references
        if "author" in df.columns:
            standardized_df["references"] = df.apply(
                lambda row: f"Author: {row.get('author', '')}, URL: {row.get('references', '')}", 
                axis=1
            )

    # Fill missing values
    standardized_df = standardized_df.fillna("")

    # Truncate long text fields to avoid CSV issues
    for col in ["content", "description", "recommendations"]:
        standardized_df[col] = standardized_df[col].apply(
            lambda x: (x[:10000] + "...") if isinstance(x, str) and len(x) > 10000 else x
        )

    return standardized_df

def clean_text_fields(df):
    """
    Clean text fields in the DataFrame.
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Define text columns to clean
    text_columns = ["topic", "description", "content", "symptoms", "treatment", 
                   "recommendations", "statistics", "references"]
    
    for col in text_columns:
        if col in cleaned_df.columns:
            # Replace newlines with spaces
            cleaned_df[col] = cleaned_df[col].apply(
                lambda x: re.sub(r'\s+', ' ', str(x)) if isinstance(x, str) else x
            )
            
            # Remove any control characters
            cleaned_df[col] = cleaned_df[col].apply(
                lambda x: re.sub(r'[\x00-\x1F\x7F]', '', str(x)) if isinstance(x, str) else x
            )
            
            # Strip whitespace
            cleaned_df[col] = cleaned_df[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
    
    return cleaned_df

def categorize_women_health_topics(df):
    """
    Categorize topics related to women's health.
    
    Args:
        df: DataFrame to categorize
    
    Returns:
        DataFrame with added category information
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    categorized_df = df.copy()
    
    # Define category keywords
    category_keywords = {
        "Reproductive Health": ["pregnancy", "birth", "fertility", "contraception", 
                               "menstruation", "period", "menstrual", "ovulation", 
                               "conception", "ivf", "reproductive"],
        
        "Gynecological Conditions": ["endometriosis", "pcos", "fibroids", "ovarian", 
                                    "cervical", "uterine", "vaginal", "vulvar", 
                                    "gynecological", "pelvic"],
        
        "Breast Health": ["breast", "mammogram", "mastectomy", "lumpectomy"],
        
        "Menopause": ["menopause", "perimenopause", "postmenopause", "hot flash", 
                     "hot flush", "night sweat", "hormone replacement", "hrt"],
        
        "Sexual Health": ["sexual", "libido", "std", "sti", "hpv", "herpes", 
                         "contraceptive", "birth control"],
        
        "Mental Health": ["depression", "anxiety", "stress", "postpartum depression", 
                         "mental health", "emotional health", "mood"],
        
        "Preventive Care": ["screening", "prevention", "vaccine", "pap smear", 
                           "mammogram", "bone density", "preventive"],
        
        "Chronic Conditions": ["autoimmune", "thyroid", "osteoporosis", "heart disease", 
                              "diabetes", "chronic fatigue", "fibromyalgia", "lupus"],
        
        "Healthcare Access": ["access", "insurance", "cost", "barrier", "disparity", 
                             "inequality", "discrimination", "bias"]
    }
    
    # Function to determine category based on text
    def determine_category(row):
        # If already has a specific category, keep it
        if row["category"] and row["category"] != "Health Articles" and row["category"] != "General":
            return row["category"]
        
        # Check topic and content for keywords
        text_to_check = f"{row['topic']} {row['description']} {row['content']}".lower()
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_to_check for keyword in keywords):
                return category
        
        # If demographics is Women but no specific category found
        if row["demographics"] == "Women":
            return "Women's General Health"
        
        return row["category"]
    
    # Apply categorization
    categorized_df["category"] = categorized_df.apply(determine_category, axis=1)
    
    return categorized_df

def integrate_data():
    """
    Integrate data from all sources.
    
    Returns:
        Integrated DataFrame
    """
    log_message("Starting data integration process")

    # Define all sources
    sources = [
        "CDC Open Data",
        "NIA Talking with Your Doctor Worksheets",
        "NHS Inform",
        "Pew Research Reports",
        "ResearchGate Medical Studies",
        "Harvard Health Blog"
    ]

    all_data = []

    # Process each source
    for source in sources:
        log_message(f"Processing {source}")

        # Load the processed data
        df = load_processed_data(source)

        if not df.empty:
            # Standardize the DataFrame
            standardized_df = standardize_dataframe(df, source)

            # Add to the combined data
            all_data.append(standardized_df)

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        log_message(f"Combined data has {len(combined_df)} records")

        # Clean text fields
        cleaned_df = clean_text_fields(combined_df)

        return categorize_women_health_topics(cleaned_df)
    else:
        log_message("No data to integrate")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

def main():
    """Main function to integrate all data sources."""
    log_message("Starting women's health data integration process")

    # Integrate all data
    integrated_df = integrate_data()

    if not integrated_df.empty:
        _extracted_from_main_10(integrated_df)
    else:
        log_message("No data was integrated")


# TODO Rename this here and in `main`
def _extracted_from_main_10(integrated_df):
    # Save integrated data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    integrated_file = os.path.join(INTEGRATED_DATA_DIR, f"womens_health_integrated_data_{timestamp}.csv")
    integrated_df.to_csv(integrated_file, index=False, encoding='utf-8')
    log_message(f"Saved integrated data to {integrated_file}")

    # Also save a copy with a standard name for easy access
    standard_file = os.path.join(INTEGRATED_DATA_DIR, "womens_health_integrated_data.csv")
    integrated_df.to_csv(standard_file, index=False, encoding='utf-8')
    log_message(f"Saved integrated data to {standard_file}")

    # Display summary
    log_message(f"Integrated {len(integrated_df)} total records from all sources")
    log_message(f"Data sources: {', '.join(integrated_df['source'].unique())}")
    log_message(f"Categories: {', '.join(integrated_df['category'].unique())}")
    log_message(f"Demographics: {', '.join(integrated_df['demographics'].unique())}")

if __name__ == "__main__":
    main()
