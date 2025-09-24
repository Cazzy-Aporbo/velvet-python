"""
pricing_production_framework.py

Production-level pricing algorithm framework with checks and balances.
Includes:
- Data validation
- Demand modeling (linear and non-linear)
- Competitor influence handling
- Dynamic pricing optimization
- Revenue forecasting
- Logging for sanity checks
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import logging

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# -----------------------------
# 1. Data Generation & Validation
# -----------------------------
def generate_sample_data(n=200):
    np.random.seed(42)
    base_price = np.random.uniform(10, 100, n)
    competitor_price = base_price + np.random.uniform(-5, 5, n)
    marketing_spend = np.random.uniform(1000, 10000, n)

    # Demand simulation
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
        logging.warning("Negative values detected, check your data.")

    logging.info("Data validation passed.")


# -----------------------------
# 2. Demand Model
# -----------------------------
def build_demand_model(data, degree=2):
    """
    Builds a demand model using linear regression with polynomial features.
    """
    X = data[['base_price', 'competitor_price', 'marketing_spend']]
    y = data['demand']
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)

    logging.info("Demand model trained successfully.")
    return model, poly

def predict_demand(model, poly, base_price, competitor_price, marketing_spend):
    features = pd.DataFrame([[base_price, competitor_price, marketing_spend]],
                            columns=['base_price', 'competitor_price', 'marketing_spend'])
    features_poly = poly.transform(features)
    predicted_demand = model.predict(features_poly)[0]
    
    if predicted_demand < 0:
        logging.warning("Predicted demand negative, resetting to 0.")
        predicted_demand = 0
    
    return predicted_demand


# -----------------------------
# 3. Dynamic Pricing Optimization
# -----------------------------
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

    logging.info("Optimal price calculated: $%.2f with expected revenue: $%.2f",
                 best_price, max_revenue)
    return best_price, max_revenue


# -----------------------------
# 4. Revenue Forecasting
# -----------------------------
def forecast_revenue(model, poly, price, competitor_price, marketing_spend, units=1):
    demand = predict_demand(model, poly, price, competitor_price, marketing_spend)
    revenue = demand * price * units
    logging.info("Forecasted revenue for %d units: $%.2f", units, revenue)
    return revenue


# -----------------------------
# 5. Main Execution
# -----------------------------
if __name__ == "__main__":
    # Step 1: Generate data
    data = generate_sample_data(n=300)
    print(data.head())

    # Step 2: Train demand model
    model, poly = build_demand_model(data)

    # Step 3: Compute optimal price for given conditions
    competitor_price = 55
    marketing_spend = 7000
    price, revenue = optimal_price(model, poly, competitor_price, marketing_spend)
    
    # Step 4: Forecast revenue
    forecasted_revenue = forecast_revenue(model, poly, price, competitor_price, marketing_spend, units=1)
    
    print(f"\nOptimal Price: ${price:.2f}")
    print(f"Expected Revenue: ${revenue:.2f}")
    print(f"Forecasted Revenue (1 unit scenario): ${forecasted_revenue:.2f}")