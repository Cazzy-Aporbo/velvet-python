"""
Lead Data Scientist Appliance Repair Portfolio Project
- End-to-end pipeline: generate, load, clean, analyze, model, visualize, export
- Demonstrates professional-level data science skills
"""

import pandas as pd
import numpy as np
import random
import faker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from datetime import datetime, timedelta

# -----------------------------
# 1. Generate Synthetic Appliance Repair Data
# -----------------------------
fake = faker.Faker()
num_customers = 200
years = 5
records_per_customer = 10  # Avg service visits per customer

appliance_types = ["Refrigerator", "Washer", "Dryer", "Oven", "Dishwasher", "Microwave", "AC Unit"]
issues = ["Not working", "Leaking", "Noisy", "Electrical problem", "Broken part", "Temperature issue"]
technicians = ["John", "Mike", "Sara", "Emma", "Alex", "Tom"]

records = []

for _ in range(num_customers):
    first_name = fake.first_name()
    last_name = fake.last_name()
    email = fake.email()
    phone = fake.phone_number()
    address = fake.address().replace("\n", ", ")
    
    num_services = random.randint(3, records_per_customer)
    
    for _ in range(num_services):
        appliance = random.choice(appliance_types)
        issue = random.choice(issues)
        technician = random.choice(technicians)
        service_date = datetime.today() - timedelta(days=random.randint(0, years*365))
        cost = round(random.uniform(50, 1000), 2)
        parts_replaced = random.choice([0,1,2,3])
        duration_hours = round(random.uniform(0.5, 5.0),1)
        
        records.append({
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": phone,
            "address": address,
            "appliance_type": appliance,
            "issue": issue,
            "technician": technician,
            "service_date": service_date.date(),
            "cost": cost,
            "parts_replaced": parts_replaced,
            "duration_hours": duration_hours
        })

df = pd.DataFrame(records)

# Introduce messy data
for col in df.columns:
    df.loc[df.sample(frac=0.05).index, col] = np.nan  # 5% missing
df = pd.concat([df, df.sample(frac=0.02)], ignore_index=True)  # 2% duplicates

# Export raw data
df.to_csv("synthetic_appliance_repair_raw.csv", index=False)
print("Synthetic raw appliance repair data generated")

# -----------------------------
# 2. Read Back and Clean
# -----------------------------
df = pd.read_csv("synthetic_appliance_repair_raw.csv")
# Fill missing numeric with median
for col in ["cost","parts_replaced","duration_hours"]:
    df[col].fillna(df[col].median(), inplace=True)
# Fill missing categorical with mode
for col in ["technician","appliance_type","issue"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

# -----------------------------
# 3. Exploratory Data Analysis
# -----------------------------
print(df.describe())
sns.histplot(df["cost"], bins=30, kde=True)
plt.title("Service Cost Distribution")
plt.show()

sns.countplot(x="appliance_type", data=df)
plt.title("Appliance Types Serviced")
plt.show()

sns.boxplot(x="technician", y="duration_hours", data=df)
plt.title("Service Duration by Technician")
plt.show()

# -----------------------------
# 4. Feature Engineering
# -----------------------------
# Revenue per customer
df["customer_id"] = df["email"]
revenue_per_customer = df.groupby("customer_id")["cost"].sum().reset_index().rename(columns={"cost":"total_revenue"})
df = df.merge(revenue_per_customer, on="customer_id")

# High-cost service flag
df["high_cost"] = (df["cost"] > df["cost"].median()).astype(int)

# -----------------------------
# 5. Predictive Modeling
# -----------------------------
# Regression: predict cost
features = ["parts_replaced","duration_hours"]
X = df[features]
y_reg = df["cost"]
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)
print("\nRegression R^2:", r2_score(y_test, y_pred))
print("Regression RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# Classification: high-cost service
y_clf = df["high_cost"]
clf_model = LogisticRegression()
clf_model.fit(X_train, y_clf)
y_pred_clf = clf_model.predict(X_test)
print("\nClassification Accuracy:", accuracy_score(y_clf, y_pred_clf))
print(classification_report(y_clf, y_pred_clf))

# -----------------------------
# 6. Business Insights
# -----------------------------
# Top 5 technicians by revenue
top_tech = df.groupby("technician")["cost"].sum().sort_values(ascending=False).head(5)
print("\nTop 5 Technicians by Revenue:\n", top_tech)

# Most common issues
top_issues = df["issue"].value_counts().head(5)
print("\nTop 5 Most Common Issues:\n", top_issues)

# -----------------------------
# 7. Export Cleaned & Analyzed Data
# -----------------------------
df.to_csv("synthetic_appliance_repair_cleaned.csv", index=False)
print("\nCleaned and analyzed appliance repair dataset exported")