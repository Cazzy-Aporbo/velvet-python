"""
invoice_processing_framework.py

Production-ready framework to automate invoice processing using AI Builder
Extracts invoice data, validates fields, and generates structured output.

Author: ChatGPT
"""

import logging
import pandas as pd
import requests

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# -----------------------------
# 1. Fetch invoice via API or file
# -----------------------------
def fetch_invoice(file_path=None, api_url=None, invoice_id=None):
    """
    Fetch an invoice from local file or via API
    """
    if file_path:
        with open(file_path, 'rb') as f:
            content = f.read()
        logging.info("Invoice loaded from file: %s", file_path)
        return content
    elif api_url and invoice_id:
        try:
            response = requests.get(f"{api_url}?invoice_id={invoice_id}", timeout=5)
            response.raise_for_status()
            logging.info("Invoice fetched via API: %s", invoice_id)
            return response.content
        except Exception as e:
            logging.error("Error fetching invoice: %s", e)
            return None
    else:
        raise ValueError("Either file_path or api_url + invoice_id must be provided")

# -----------------------------
# 2. Extract invoice fields using AI Builder
# -----------------------------
def extract_invoice_fields(invoice_bytes, ai_builder_api_url, api_key):
    """
    Sends invoice bytes to AI Builder Invoice Processing model and returns extracted fields
    """
    try:
        headers = {'Ocp-Apim-Subscription-Key': api_key}
        files = {'file': invoice_bytes}
        response = requests.post(ai_builder_api_url, headers=headers, files=files)
        response.raise_for_status()
        data = response.json()
        logging.info("Invoice fields extracted successfully")
        return data.get('fields', {})
    except Exception as e:
        logging.error("Error extracting invoice fields: %s", e)
        return {}

# -----------------------------
# 3. Validate extracted fields
# -----------------------------
def validate_invoice_fields(fields):
    """
    Validates mandatory fields like vendor, invoice_date, total_amount
    """
    required_fields = ['vendor', 'invoice_date', 'total_amount']
    missing = [f for f in required_fields if f not in fields or not fields[f]]
    if missing:
        logging.warning("Missing mandatory invoice fields: %s", missing)
    else:
        logging.info("All mandatory fields present")
    return missing

# -----------------------------
# 4. Transform and prepare data for downstream system
# -----------------------------
def transform_invoice_data(fields):
    """
    Convert fields to structured DataFrame for accounting system or database
    """
    df = pd.DataFrame([{
        'vendor': fields.get('vendor'),
        'invoice_date': fields.get('invoice_date'),
        'due_date': fields.get('due_date'),
        'invoice_number': fields.get('invoice_number'),
        'total_amount': fields.get('total_amount'),
        'currency': fields.get('currency')
    }])
    return df

# -----------------------------
# 5. Main execution
# -----------------------------
if __name__ == "__main__":
    INVOICE_FILE = "example_invoice.pdf"  # Replace with actual file
    AI_BUILDER_API_URL = "https://your-ai-builder-endpoint/invoiceprocessing"
    API_KEY = "YOUR_API_KEY"

    invoice_bytes = fetch_invoice(file_path=INVOICE_FILE)
    if invoice_bytes:
        fields = extract_invoice_fields(invoice_bytes, AI_BUILDER_API_URL, API_KEY)
        missing_fields = validate_invoice_fields(fields)
        if not missing_fields:
            invoice_df = transform_invoice_data(fields)
            print("\nExtracted Invoice Data:")
            print(invoice_df)
        else:
            logging.error("Invoice processing incomplete due to missing fields: %s", missing_fields)