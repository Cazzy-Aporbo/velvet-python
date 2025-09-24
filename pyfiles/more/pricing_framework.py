"""
pricing_framework.py

A comprehensive framework for understanding and experimenting with pricing algorithms.
Includes demand modeling, price elasticity, competitor influence, and dynamic pricing.

Author: ChatGPT
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# -----------------------------
# 1. Sample Data Generation
# -----------------------------
def generate_sample_data(n=100):
    """
    Generates sample data with base_price, competitor_price, marketing_spend, and demand.
    """
    np.random.seed(42)
    base_price = np.random.uniform(10, 100, n)
    competitor_price = base_price + np.random.uniform(-5, 5, n)
    marketing_spend = np.random.uniform(1000, 10000, n)
    
    # Simulate demand using price elasticity
    demand = (200 - 1.5 * base_price + 0.5 * competitor_price +
              0.02 * marketing_spend + np.random.normal(0, 5, n))
    demand = np.maximum(demand, 0)  # Demand can't be negative
    
    data = pd.DataFrame({
        'base_price': base_price,
        'competitor_price': competitor_price,
        'marketing_spend': marketing_spend,
        'demand': demand
    })
    return data

# -----------------------------
# 2. Price Elasticity Model
# -----------------------------
def price_elasticity_model(data):
    """
    Fits a linear regression model to understand price elasticity.
    """
    X = data[['base_price', 'competitor_price', 'marketing_spend']]
    y = data['demand']
    
    # Polynomial features to capture non-linear effects
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    print("Price Elasticity Model Coefficients:")
    for feature, coef in zip(poly.get_feature_names_out(X.columns), model.coef_):
        print(f"{feature}: {coef:.4f}")
    
    return model, poly

# -----------------------------
# 3. Dynamic Pricing Strategy
# -----------------------------
def optimal_price(model, poly, competitor_price, marketing_spend, price_range=(10, 100)):
    """
    Calculates optimal price to maximize revenue given competitor price and marketing spend.
    """
    best_price = None
    max_revenue = -np.inf
    
    for price in np.linspace(price_range[0], price_range[1], 100):
        features = pd.DataFrame([[price, competitor_price, marketing_spend]],
                                columns=['base_price', 'competitor_price', 'marketing_spend'])
        features_poly = poly.transform(features)
        predicted_demand = model.predict(features_poly)[0]
        revenue = predicted_demand * price
        
        if revenue > max_revenue:
            max_revenue = revenue
            best_price = price
    
    return best_price, max_revenue

# -----------------------------
# 4. Main Execution
# -----------------------------
if __name__ == "__main__":
    # Generate sample dataset
    data = generate_sample_data(n=200)
    print("Sample Data:")
    print(data.head())

    # Fit pricing model
    model, poly = price_elasticity_model(data)
    
    # Example: Compute optimal price for given competitor price and marketing spend
    competitor_price = 50
    marketing_spend = 5000
    price, revenue = optimal_price(model, poly, competitor_price, marketing_spend)
    
    print(f"\nOptimal Price: ${price:.2f}")
    print(f"Expected Revenue: ${revenue:.2f}")