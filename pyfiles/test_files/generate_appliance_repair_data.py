import pandas as pd
import numpy as np
import random
import faker
from datetime import datetime, timedelta

fake = faker.Faker()
num_customers = 200
years = 5
records_per_customer = 10  # Avg number of service visits per customer

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
        # Random service date in last 5 years
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
# 5% missing values
for col in df.columns:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

# 2% duplicate rows
df = pd.concat([df, df.sample(frac=0.02)], ignore_index=True)

# Random inconsistent entries
df.loc[df.sample(frac=0.03).index, "appliance_type"] = df["appliance_type"].sample(frac=0.03).values + " "

# Shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

# Save CSV
df.to_csv("synthetic_appliance_repair_data.csv", index=False)
print("Synthetic appliance repair dataset generated: synthetic_appliance_repair_data.csv")