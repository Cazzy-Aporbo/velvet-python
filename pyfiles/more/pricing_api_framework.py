"""
pricing_api_framework.py

Production level pricing algorithm with API integration for competitor monitoring
Includes:
API call to fetch competitor prices
Data validation and logging
Demand modeling linear and polynomial
Dynamic pricing optimization
Revenue forecasting

Author: ChatGPT
"""

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# Fetch Competitor Prices
def fetch_competitor_price(api_url, product_id):
    """
    Fetch competitor price from an API
    Example API response: {"product_id": "123", "competitor_price": 49.99}
    """
    try:
        response = requests.get(f"{api_url}?product_id={product_id}", timeout=5)
        response.raise_for_status()
        data = response.json()
        price = data.get('competitor_price')
        if price is None or price < 0:
            raise ValueError(f"Invalid price received from API: {price}")
        logging.info("Fetched competitor price %.2f for product %s", price, product_id)
        return price
    except Exception as e:
        logging.error("Error fetching competitor price: %s", e)
        return None

# Data Generation and Validation
def generate_sample_data(n=200):
    np.random.seed(42)
    base_price = np.random.uniform(10, 100, n)
    competitor_price = base_price + np.random.uniform(-5, 5, n)
    marketing_spend = np.random.uniform(1000, 10000, n)

    demand = (200 - 1.5 * base_price + 0.5 * competitor_price +
              0.02 * marketing_spend + np.random.normal(0, 5, n))
    demand = np.maximum(demand, 0)

    data = pd.DataFrame({
        'base_price': base_price,
        'competitor_price': competitor_price,
        'marketing_spend': marketing_spend,
        'demand': demand
    })

    logging.info("Sample data generated with %d rows", len(data))
    validate_data(data)
    return data

def validate_data(data):
    required_columns = ['base_price', 'competitor_price', 'marketing_spend', 'demand']
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if (data[['base_price', 'competitor_price', 'marketing_spend', 'demand']] < 0).any().any():
        logging.warning("Negative values detected, check your data")

    logging.info("Data validation passed")

# Demand Model
def build_demand_model(data, degree=2):
    X = data[['base_price', 'competitor_price', 'marketing_spend']]
    y = data['demand']

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    logging.info("Demand model trained successfully")
    return model, poly

def predict_demand(model, poly, base_price, competitor_price, marketing_spend):
    features = pd.DataFrame([[base_price, competitor_price, marketing_spend]],
                            columns=['base_price', 'competitor_price', 'marketing_spend'])
    features_poly = poly.transform(features)
    predicted_demand = model.predict(features_poly)[0]

    if predicted_demand < 0:
        logging.warning("Predicted demand negative, resetting to 0")
        predicted_demand = 0

    return predicted_demand

# Dynamic Pricing Optimization
def optimal_price(model, poly, competitor_price, marketing_spend,
                  price_range=(10, 100), steps=100):
    best_price = None
    max_revenue = -np.inf

    for price in np.linspace(price_range[0], price_range[1], steps):
        demand = predict_demand(model, poly, price, competitor_price, marketing_spend)
        revenue = demand * price
        if revenue > max_revenue:
            max_revenue = revenue
            best_price = price

    logging.info("Optimal price %.2f Expected revenue %.2f", best_price, max_revenue)
    return best_price, max_revenue

# Revenue Forecasting
def forecast_revenue(model, poly, price, competitor_price, marketing_spend, units=1):
    demand = predict_demand(model, poly, price, competitor_price, marketing_spend)
    revenue = demand * price * units
    logging.info("Forecasted revenue for %d units %.2f", units, revenue)
    return revenue

# Main Execution
if __name__ == "__main__":
    API_URL = "https://example.com/api/competitor_price"  # Replace with real API
    PRODUCT_ID = "123"

    competitor_price = fetch_competitor_price(API_URL, PRODUCT_ID)
    if competitor_price is None:
        competitor_price = 50

    data = generate_sample_data(n=300)
    model, poly = build_demand_model(data)

    marketing_spend = 7000
    price, revenue = optimal_price(model, poly, competitor_price, marketing_spend)
    forecasted_revenue = forecast_revenue(model, poly, price, competitor_price, marketing_spend, units=1)

    print(f"\nOptimal Price: ${price:.2f}")
    print(f"Expected Revenue: ${revenue:.2f}")
    print(f"Forecasted Revenue 1 unit scenario: ${forecasted_revenue:.2f}")