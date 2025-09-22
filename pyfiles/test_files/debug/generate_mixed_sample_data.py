import pandas as pd
import numpy as np
import random
import faker

fake = faker.Faker()
num_rows = 500

# Generate safe fields
safe_data = {
    "department": [random.choice(["HR", "Finance", "IT", "Marketing"]) for _ in range(num_rows)],
    "score": [round(random.uniform(0, 100), 2) for _ in range(num_rows)],
    "years_experience": [random.randint(0, 40) for _ in range(num_rows)],
    "project_count": [random.randint(0, 20) for _ in range(num_rows)]
}

# Generate PHI-risky fields (HIPAA-like)
phi_data = {
    "first_name": [fake.first_name() for _ in range(num_rows)],
    "last_name": [fake.last_name() for _ in range(num_rows)],
    "dob": [fake.date_of_birth(minimum_age=18, maximum_age=90) for _ in range(num_rows)],
    "email": [fake.email() for _ in range(num_rows)],
    "phone": [fake.phone_number() for _ in range(num_rows)],
    "ssn": [f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}" for _ in range(num_rows)],
    "address": [fake.address().replace("\n", ", ") for _ in range(num_rows)],
    "patient_id": [f"MRN{random.randint(1000,9999)}" for _ in range(num_rows)]
}

# Combine safe + risky
data = {**safe_data, **phi_data}

df = pd.DataFrame(data)

# Introduce some missing values randomly
for col in df.columns:
    df.loc[df.sample(frac=0.05).index, col] = np.nan  # 5% missing

# Save to CSV
df.to_csv("mixed_sample_data.csv", index=False)
print("Mixed sample data generated: mixed_sample_data.csv")