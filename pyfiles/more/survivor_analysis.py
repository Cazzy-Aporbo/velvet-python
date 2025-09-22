import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime

# Dictionary of season names and their corresponding Google Sheet CSV export links
season_links = {
    "Borneo S1": "https://docs.google.com/spreadsheets/d/1EXAMPLE_BORNEO/export?format=csv",
    "Australia S2": "https://docs.google.com/spreadsheets/d/1EXAMPLE_AUSTRALIA/export?format=csv",
    "Africa S3": "https://docs.google.com/spreadsheets/d/1EXAMPLE_AFRICA/export?format=csv",
    "Marquesas S4": "https://docs.google.com/spreadsheets/d/1EXAMPLE_MARQUESAS/export?format=csv",
    "Thailand S5": "https://docs.google.com/spreadsheets/d/1EXAMPLE_THAILAND/export?format=csv",
    "Amazon S6": "https://docs.google.com/spreadsheets/d/1EXAMPLE_AMAZON/export?format=csv",
    "Pearl Islands S7": "https://docs.google.com/spreadsheets/d/1EXAMPLE_PEARL_ISLANDS/export?format=csv",
    "All Stars S8": "https://docs.google.com/spreadsheets/d/1EXAMPLE_ALL_STARS/export?format=csv",
    "Vanuatu S9": "https://docs.google.com/spreadsheets/d/1EXAMPLE_VANUATU/export?format=csv",
    "Palau S10": "https://docs.google.com/spreadsheets/d/1EXAMPLE_PALAU/export?format=csv",
    "Guatemala S11": "https://docs.google.com/spreadsheets/d/1EXAMPLE_GUATEMALA/export?format=csv",
    "Panama S12": "https://docs.google.com/spreadsheets/d/1EXAMPLE_PANAMA/export?format=csv",
    "Cook Islands S13": "https://docs.google.com/spreadsheets/d/1EXAMPLE_COOK_ISLANDS/export?format=csv",
    "Fiji S14": "https://docs.google.com/spreadsheets/d/1EXAMPLE_FIJI/export?format=csv",
    "China S15": "https://docs.google.com/spreadsheets/d/1EXAMPLE_CHINA/export?format=csv",
    "Micronesia S16": "https://docs.google.com/spreadsheets/d/1EXAMPLE_MICRONESIA/export?format=csv",
    "Gabon S17": "https://docs.google.com/spreadsheets/d/1EXAMPLE_GABON/export?format=csv",
    "Tocantins S18": "https://docs.google.com/spreadsheets/d/1EXAMPLE_TOCANTINS/export?format=csv",
    "Samoa S19": "https://docs.google.com/spreadsheets/d/1EXAMPLE_SAMOA/export?format=csv",
    "Heroes vs. Villains S20": "https://docs.google.com/spreadsheets/d/1EXAMPLE_HEROES_VS_VILLAINS/export?format=csv",
    "Nicaragua S21": "https://docs.google.com/spreadsheets/d/1EXAMPLE_NICARAGUA/export?format=csv",
    "Redemption Island S22": "https://docs.google.com/spreadsheets/d/1EXAMPLE_REDEMPTION_ISLAND/export?format=csv",
    "South Pacific S23": "https://docs.google.com/spreadsheets/d/1EXAMPLE_SOUTH_PACIFIC/export?format=csv",
    "One World S24": "https://docs.google.com/spreadsheets/d/1EXAMPLE_ONE_WORLD/export?format=csv",
    "Philippines S25": "https://docs.google.com/spreadsheets/d/1EXAMPLE_PHILIPPINES/export?format=csv",
    "Caramoan S26": "https://docs.google.com/spreadsheets/d/1EXAMPLE_CARAMOAN/export?format=csv",
    "Cagayan S27": "https://docs.google.com/spreadsheets/d/1EXAMPLE_CAGAYAN/export?format=csv",
}

# Load and merge all seasons' data
all_data = []
for season_name, url in season_links.items():
    df = pd.read_csv(url)
    df['Season'] = season_name
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# Clean numeric columns
numeric_cols = ['ChW%', 'TC%', 'JV%', 'SurvSc', 'SurvAv']
for col in numeric_cols:
    if col in df_all.columns:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

# Fill missing values with 0
df_all[numeric_cols] = df_all[numeric_cols].fillna(0)

# Top performers
top10_survsc = df_all.sort_values('SurvSc', ascending=False).head(10)
top10_survav = df_all.sort_values('SurvAv', ascending=False).head(10)

# Interactive color-coded plots
fig_survsc = px.scatter(
    df_all,
    x="ChW%",
    y="SurvSc",
    color="SurvSc",
    hover_data=["Contestant", "Season", "TC%", "JV%"],
    color_continuous_scale="Viridis",
    title="Survival Score vs Challenge Win % (All Seasons)"
)

fig_survav = px.scatter(
    df_all,
    x="TC%",
    y="SurvAv",
    color="SurvAv",
    hover_data=["Contestant", "Season", "ChW%", "JV%"],
    color_continuous_scale="Plasma",
    title="Survival Average vs Tribal Council % (All Seasons)"
)

# Save plots to HTML
pio.write_html(fig_survsc, file="survsc_plot.html", auto_open=False)
pio.write_html(fig_survav, file="survav_plot.html", auto_open=False)

# Generate single dashboard HTML
html_content = f"""
<html>
<head><title>Survivor All Seasons Dashboard</title></head>
<body>
<h1>Survivor: All Seasons Advanced Analysis</h1>
<p>Report generated: {datetime.now()}</p>

<h2>Top 10 Contestants by Survival Score</h2>
{top10_survsc[['Contestant', 'Season', 'ChW%', 'TC%', 'JV%', 'SurvSc']].to_html(index=False)}

<h2>Top 10 Contestants by Survival Average</h2>
{top10_survav[['Contestant', 'Season', 'ChW%', 'TC%', 'JV%', 'SurvAv']].to_html(index=False)}

<h2>Survival Score vs Challenge Win %</h2>
<iframe src="survsc_plot.html" width="100%" height="600"></iframe>

<h2>Survival Average vs Tribal Council %</h2>
<iframe src="survav_plot.html" width="100%" height="600"></iframe>

</body>
</html>
"""

with open("survivor_all_seasons_dashboard.html", "w") as f:
    f.write(html_content)

print("Interactive Survivor dashboard generated: survivor_all_seasons_dashboard.html")