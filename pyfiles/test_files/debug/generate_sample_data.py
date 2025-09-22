import pandas as pd
import numpy as np
import random
import faker

# Initialize Faker for realistic names, emails, addresses
fake = faker.Faker()

num_rows = 500

data = {
    "first_name": [fake.first_name() for _ in range(num_rows)],
    "last_name": [fake.last_name() for _ in range(num_rows)],
    "dob": [fake.date_of_birth(minimum_age=18, maximum_age=90) for _ in range(num_rows)],
    "email": [fake.email() for _ in range(num_rows)],
    "phone": [fake.phone_number() for _ in range(num_rows)],
    "ssn": [f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}" for _ in range(num_rows)],
    "address": [fake.address().replace("\n", ", ") for _ in range(num_rows)],
    "age": [random.randint(18, 90) for _ in range(num_rows)],
    "income": [round(random.uniform(20000,150000), 2) for _ in range(num_rows)],
    "department": [random.choice(["HR", "Finance", "IT", "Marketing"]) for _ in range(num_rows)],
    "score": [round(random.uniform(0,100),2) for _ in range(num_rows)]
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("sample_data.csv", index=False)
print("Sample data generated: sample_data.csv")