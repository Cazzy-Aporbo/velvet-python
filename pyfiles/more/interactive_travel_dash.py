import pandas as pd
import random
from datetime import datetime
import plotly.express as px
import plotly.io as pio

# Simulate flight data (replace with real API)
destinations = ["Paris", "Tokyo", "New York", "Dubai", "Sydney", "Rome", "Barcelona", "Bangkok"]
airlines = ["Delta", "Emirates", "Qatar Airways", "Air France", "ANA"]

flights = pd.DataFrame([{
    "destination": random.choice(destinations),
    "airline": random.choice(airlines),
    "price": round(random.uniform(200, 1500), 2),
    "duration_hours": random.randint(6, 20)
} for _ in range(300)])

# Simulate car rental data (replace with real API)
car_cities = ["Paris", "Tokyo", "New York", "Dubai", "Sydney", "Rome", "Barcelona", "Bangkok"]
car_types = ["Compact", "SUV", "Luxury", "Van"]
companies = ["Hertz", "Avis", "Enterprise", "Sixt"]

cars = pd.DataFrame([{
    "city": random.choice(car_cities),
    "car_type": random.choice(car_types),
    "company": random.choice(companies),
    "price_per_day": round(random.uniform(25, 250), 2)
} for _ in range(300)])

# Flight: cheapest destinations by color-coded price
fig_flight = px.scatter(
    flights,
    x="destination",
    y="price",
    color="price",
    size="duration_hours",
    hover_data=["airline", "duration_hours"],
    color_continuous_scale="Viridis",
    title="Flights: Cheapest Destinations & Prices"
)
pio.write_html(fig_flight, file="flights_dashboard.html", auto_open=False)

# Car: cheapest rentals by city, color-coded
fig_car = px.scatter(
    cars,
    x="city",
    y="price_per_day",
    color="price_per_day",
    size="price_per_day",
    hover_data=["car_type", "company"],
    color_continuous_scale="Plasma",
    title="Car Rentals: Cheapest Options by City"
)
pio.write_html(fig_car, file="cars_dashboard.html", auto_open=False)

# Merge flights & cars into a single interactive HTML
html_content = f"""
<html>
<head><title>Interactive Travel Dashboard</title></head>
<body>
<h1>Interactive Travel Dashboard</h1>
<p>Report generated: {datetime.now()}</p>
<h2>Flights</h2>
<iframe src="flights_dashboard.html" width="100%" height="600"></iframe>
<h2>Car Rentals</h2>
<iframe src="cars_dashboard.html" width="100%" height="600"></iframe>
</body>
</html>
"""

with open("interactive_travel_report.html", "w") as f:
    f.write(html_content)

print("Interactive travel dashboard generated: interactive_travel_report.html")