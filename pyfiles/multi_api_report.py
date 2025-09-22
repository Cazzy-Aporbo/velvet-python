import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

fake = None  # not needed here but kept if extended

try:
    covid_url = "https://api.covid19api.com/dayone/country/us/status/confirmed/live"
    covid_data = requests.get(covid_url).json()
    df_covid = pd.DataFrame(covid_data)
    print("COVID API loaded:", df_covid.shape)
except Exception as e:
    print("Error loading COVID API:", e)
    df_covid = pd.DataFrame()

try:
    users_url = "https://randomuser.me/api/?results=100"
    users_data = requests.get(users_url).json()["results"]
    df_users = pd.json_normalize(users_data)
    print("Random users API loaded:", df_users.shape)
except Exception as e:
    print("Error loading users API:", e)
    df_users = pd.DataFrame()

if not df_covid.empty:
    df_covid = df_covid[["Country","Province","City","Cases","Date"]]
    df_covid["Date"] = pd.to_datetime(df_covid["Date"])
    df_covid["Cases"].fillna(0, inplace=True)

if not df_users.empty:
    df_users = df_users[["name.first","name.last","location.city","location.state","email","dob.date"]]
    df_users.rename(columns={
        "name.first":"first_name",
        "name.last":"last_name",
        "location.city":"city",
        "location.state":"state",
        "dob.date":"dob"
    }, inplace=True)
    df_users["dob"] = pd.to_datetime(df_users["dob"])

if not df_users.empty:
    city_counts = df_users.groupby("city").size().reset_index(name="user_count")

analysis_results = {}
if not df_covid.empty:
    latest = df_covid.groupby("Province")["Cases"].max().sort_values(ascending=False).head(5)
    analysis_results["top5_covid_provinces"] = latest
if not df_users.empty:
    top_cities = city_counts.sort_values("user_count", ascending=False).head(5)
    analysis_results["top5_user_cities"] = top_cities

plots = []
if not df_covid.empty:
    plt.figure(figsize=(10,5))
    df_covid.groupby("Date")["Cases"].sum().plot()
    plt.title("Total US COVID Cases Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.tight_layout()
    plt.savefig("covid_cases.png")
    plots.append("covid_cases.png")
    plt.close()

if not df_users.empty:
    plt.figure(figsize=(8,5))
    city_counts.sort_values("user_count", ascending=False).head(10).plot.bar(x="city", y="user_count")
    plt.title("Top 10 Cities by Random Users")
    plt.xlabel("City")
    plt.ylabel("User Count")
    plt.tight_layout()
    plt.savefig("user_city_counts.png")
    plots.append("user_city_counts.png")
    plt.close()

html_content = f"""
<html>
<head>
<title>Multi-API Data Report</title>
</head>
<body>
<h1>Multi-API Data Report</h1>
<h2>Report generated: {datetime.now()}</h2>
"""

if "top5_covid_provinces" in analysis_results:
    html_content += "<h3>Top 5 COVID Provinces (Latest Cases)</h3>"
    html_content += analysis_results["top5_covid_provinces"].to_frame().to_html()

if "top5_user_cities" in analysis_results:
    html_content += "<h3>Top 5 User Cities</h3>"
    html_content += analysis_results["top5_user_cities"].to_html()

for plot in plots:
    html_content += f'<h3>{plot}</h3>'
    html_content += f'<img src="{plot}" width="600"><br>'

html_content += "</body></html>"

with open("multi_api_report.html","w") as f:
    f.write(html_content)

print("HTML report generated: multi_api_report.html")