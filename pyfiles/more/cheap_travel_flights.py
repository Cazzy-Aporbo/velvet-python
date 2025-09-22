import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random

def get_flight_data():
    # Simulate flight data (replace with real API call)
    destinations = ["Paris", "Tokyo", "New York", "Dubai", "Sydney", "Rome", "Barcelona", "Bangkok"]
    flights = []
    for _ in range(200):
        dest = random.choice(destinations)
        price = round(random.uniform(200, 1500), 2)
        airline = random.choice(["Delta", "Emirates", "Qatar Airways", "Air France", "ANA"])
        duration = random.randint(6, 20)  # hours
        flights.append({
            "destination": dest,
            "price": price,
            "airline": airline,
            "duration_hours": duration
        })
    return pd.DataFrame(flights)

def get_car_rental_data():
    # Simulate car rental data (replace with real API call)
    cities = ["Paris", "Tokyo", "New York", "Dubai", "Sydney", "Rome", "Barcelona", "Bangkok"]
    cars = []
    for _ in range(200):
        city = random.choice(cities)
        car_type = random.choice(["Compact", "SUV", "Luxury", "Van"])
        price_per_day = round(random.uniform(25, 250), 2)
        company = random.choice(["Hertz", "Avis", "Enterprise", "Sixt"])
        cars.append({
            "city": city,
            "car_type": car_type,
            "price_per_day": price_per_day,
            "company": company
        })
    return pd.DataFrame(cars)

def plot_top(df, column, top_n=10, filename=None, title=None):
    counts = df[column].value_counts().head(top_n)
    plt.figure(figsize=(10,5))
    counts.plot.barh()
    plt.title(title if title else f"Top {column}")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()

def plot_distribution(df, column, filename=None, title=None):
    plt.figure(figsize=(10,5))
    df[column].hist(bins=30)
    plt.title(title if title else f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()

def generate_html_report(flights, cars, images, filename="cheap_travel_report.html"):
    html_content = f"""
    <html>
    <head><title>Cheap Travel Report</title></head>
    <body>
    <h1>Cheap Vacation Flights & Car Rentals</h1>
    <p>Report generated: {datetime.now()}</p>
    """
    for caption, img in images.items():
        html_content += f"<h2>{caption}</h2><img src='{img}' width='600'><br>"
    html_content += "<h2>Sample Flight Data</h2>"
    html_content += flights.head(20).to_html()
    html_content += "<h2>Sample Car Rental Data</h2>"
    html_content += cars.head(20).to_html()
    html_content += "</body></html>"
    with open(filename, "w") as f:
        f.write(html_content)
    print(f"Report generated: {filename}")

def main():
    flights = get_flight_data()
    cars = get_car_rental_data()

    plot_top(flights, "destination", filename="top_flight_destinations.png", title="Top Flight Destinations")
    plot_distribution(flights, "price", filename="flight_price_distribution.png", title="Flight Price Distribution")
    plot_top(cars, "city", filename="top_car_cities.png", title="Top Car Rental Cities")
    plot_distribution(cars, "price_per_day", filename="car_price_distribution.png", title="Car Rental Price Distribution")

    images = {
        "Top Flight Destinations": "top_flight_destinations.png",
        "Flight Price Distribution": "flight_price_distribution.png",
        "Top Car Rental Cities": "top_car_cities.png",
        "Car Rental Price Distribution": "car_price_distribution.png"
    }

    generate_html_report(flights, cars, images)

if __name__ == "__main__":
    main()