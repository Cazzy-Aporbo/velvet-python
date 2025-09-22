"""
Lead Data Scientist Portfolio Project
- End-to-end pipeline: generate, load, clean, analyze, model, visualize, export
- Demonstrates data science skills
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

# -----------------------------
# 1. Generate Synthetic Healthcare Data
# -----------------------------
fake = faker.Faker()
num_patients = 500
random.seed(42)
np.random.seed(42)

data = {
    "patient_id": [f"MRN{random.randint(10000,99999)}" for _ in range(num_patients)],
    "age": np.random.randint(18, 90, size=num_patients),
    "gender": np.random.choice(["Male","Female"], size=num_patients),
    "bmi": np.round(np.random.normal(27, 5, size=num_patients),1),
    "blood_pressure": np.random.randint(90, 180, size=num_patients),
    "cholesterol": np.random.randint(150, 300, size=num_patients),
    "diagnosis": np.random.choice(["Hypertension","Diabetes","Healthy","Obesity"], size=num_patients),
    "visits_per_year": np.random.poisson(2, size=num_patients)
}

df = pd.DataFrame(data)

# Introduce some missing/dirty data
df.loc[df.sample(frac=0.05).index, "bmi"] = np.nan
df.loc[df.sample(frac=0.03).index, "blood_pressure"] = np.nan
df.loc[df.sample(frac=0.02).index, "diagnosis"] = np.nan

# Export raw synthetic data
df.to_csv("synthetic_healthcare_raw.csv", index=False)
print("Synthetic raw data generated: synthetic_healthcare_raw.csv")

# -----------------------------
# 2. Read Back and Clean
# -----------------------------
try:
    df = pd.read_csv("synthetic_healthcare_raw.csv")
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Fill missing numeric values with median
for col in ["bmi","blood_pressure"]:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with mode
df["diagnosis"].fillna(df["diagnosis"].mode()[0], inplace=True)

# Encode gender
df["gender_encoded"] = df["gender"].map({"Male":0,"Female":1})

# -----------------------------
# 3. Exploratory Data Analysis
# -----------------------------
print("\n=== Summary Statistics ===")
print(df.describe())

sns.histplot(df["age"], kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x="diagnosis", y="bmi", data=df)
plt.title("BMI by Diagnosis")
plt.show()

# Correlation heatmap
sns.heatmap(df[["age","bmi","blood_pressure","cholesterol","visits_per_year"]].corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# -----------------------------
# 4. Feature Engineering
# -----------------------------
# Risk Score as combination of BMI, BP, Cholesterol
df["risk_score"] = 0.3*df["bmi"] + 0.4*(df["blood_pressure"]/100) + 0.3*(df["cholesterol"]/200)

# Binary classification: High Risk if risk_score > median
df["high_risk"] = (df["risk_score"] > df["risk_score"].median()).astype(int)

# -----------------------------
# 5. Predictive Modeling
# -----------------------------
# Regression: Predict visits per year
features = ["age","gender_encoded","bmi","blood_pressure","cholesterol","risk_score"]
X = df[features]
y_reg = df["visits_per_year"]
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)
print("\n=== Regression Results ===")
print(f"R^2: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")

# Classification: Predict high risk
y_clf = df["high_risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train, y_train)
y_pred_clf = clf_model.predict(X_test)
print("\n=== Classification Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_clf):.3f}")
print(classification_report(y_test, y_pred_clf))

# -----------------------------
# 6. Business Insight Example
# -----------------------------
top_risk_patients = df.sort_values("risk_score", ascending=False).head(10)
print("\nTop 10 High-Risk Patients")
print(top_risk_patients[["patient_id","risk_score","diagnosis","bmi","blood_pressure"]])

# -----------------------------
# 7. Export Cleaned & Modeled Data
# -----------------------------
df.to_csv("synthetic_healthcare_cleaned_modeled.csv", index=False)
print("\nCleaned and modeled dataset exported: synthetic_healthcare_cleaned_modeled.csv")