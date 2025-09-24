"""
contract_risk_analysis.py

Production-ready framework to analyze contracts using AI Builder
Extracts key clauses, identifies potential risks, and generates summary reports.

"""

import logging
import pandas as pd
import requests

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# -----------------------------
# 1. Fetch contract via API or file path
# -----------------------------
def fetch_contract(file_path=None, api_url=None, contract_id=None):
    """
    Fetch a contract from local file or via API
    """
    if file_path:
        with open(file_path, 'rb') as f:
            content = f.read()
        logging.info("Contract loaded from file: %s", file_path)
        return content
    elif api_url and contract_id:
        try:
            response = requests.get(f"{api_url}?contract_id={contract_id}", timeout=5)
            response.raise_for_status()
            logging.info("Contract fetched via API: %s", contract_id)
            return response.content
        except Exception as e:
            logging.error("Error fetching contract: %s", e)
            return None
    else:
        raise ValueError("Either file_path or api_url + contract_id must be provided")

# -----------------------------
# 2. Extract clauses using AI Builder Contract Processing
# -----------------------------
def extract_clauses(contract_bytes, ai_builder_api_url, api_key):
    """
    Sends contract bytes to AI Builder Contract Processing model and returns extracted entities
    """
    try:
        headers = {'Ocp-Apim-Subscription-Key': api_key}
        files = {'file': contract_bytes}
        response = requests.post(ai_builder_api_url, headers=headers, files=files)
        response.raise_for_status()
        data = response.json()
        logging.info("Clauses extracted successfully")
        return data.get('entities', [])
    except Exception as e:
        logging.error("Error extracting clauses: %s", e)
        return []

# -----------------------------
# 3. Risk classification
# -----------------------------
def classify_risk(clauses):
    """
    Simple heuristic risk classification
    Marks clauses as 'High Risk' if they contain certain keywords
    """
    high_risk_keywords = ['penalty', 'termination', 'liability', 'indemnity', 'breach']
    risk_report = []

    for clause in clauses:
        text = clause.get('text', '').lower()
        risk_level = 'Low Risk'
        for keyword in high_risk_keywords:
            if keyword in text:
                risk_level = 'High Risk'
                break
        risk_report.append({
            'clause': clause.get('name', 'Unknown'),
            'text': clause.get('text', ''),
            'risk_level': risk_level
        })
    return pd.DataFrame(risk_report)

# -----------------------------
# 4. Generate summary
# -----------------------------
def generate_summary(risk_df):
    """
    Creates a summary report counting high and low risk clauses
    """
    summary = risk_df['risk_level'].value_counts().to_dict()
    logging.info("Risk summary generated: %s", summary)
    return summary

# -----------------------------
# 5. Main execution
# -----------------------------
if __name__ == "__main__":
    # Example usage
    CONTRACT_FILE = "example_contract.pdf"  # Replace with actual file
    AI_BUILDER_API_URL = "https://your-ai-builder-endpoint/contractprocessing"
    API_KEY = "YOUR_API_KEY"

    contract_bytes = fetch_contract(file_path=CONTRACT_FILE)
    if contract_bytes:
        clauses = extract_clauses(contract_bytes, AI_BUILDER_API_URL, API_KEY)
        if clauses:
            risk_df = classify_risk(clauses)
            summary = generate_summary(risk_df)
            
            print("\nContract Risk Report:")
            print(risk_df)
            print("\nSummary:")
            print(summary)