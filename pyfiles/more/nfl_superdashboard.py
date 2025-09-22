import nfl_data_py as nfl
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import numpy as np

# === 1. Pull Multi-Year NFL Data ===
years = [2022, 2023, 2024]

pbp = nfl.import_pbp_data(years=years, downcast=True)
weekly = nfl.import_weekly_data(years=years)
seasonal = nfl.import_seasonal_data(years=years, s_type='REG')
rosters = nfl.import_seasonal_rosters(years=years)
draft = nfl.import_draft_picks(years=years)
combine = nfl.import_combine_data(years=years, positions=['QB','WR','RB','TE'])
injuries = nfl.import_injuries(years=years)
snaps = nfl.import_snap_counts(years=years)
qbr = nfl.import_qbr(years=years)

# === 2. Clean datasets ===
for df in [pbp, weekly, seasonal, rosters]:
    df = nfl.clean_nfl_data(df)

# === 3. Merge & Feature Engineering ===
df = pd.merge(seasonal, rosters, on=['player_id','season'], how='left')
df = pd.merge(df, snaps, on=['player_id','season'], how='left')
df['snap_pct'] = df['snaps'] / df['snaps'].max()

# Novel Metrics
df['efficiency'] = df['dom'].fillna(0) * df['wopr'].fillna(0)
df['air_yards_impact'] = df['ay_sh'].fillna(0) * df['yac_sh'].fillna(0)
df['player_contribution'] = (df['ppr_sh'].fillna(0) + df['efficiency']) * df['snap_pct']

# === 4. Top Performers ===
top10_contrib = df.sort_values('player_contribution', ascending=False).head(10)

# === 5. Interactive Plots ===
fig1 = px.scatter(
    df, x='ay_sh', y='yac_sh',
    color='air_yards_impact',
    size='player_contribution',
    hover_data=['player_name','team','position','season'],
    color_continuous_scale='Turbo',
    title='Air Yards vs YAC vs Air Yards Impact'
)

fig2 = px.scatter(
    df, x='snap_pct', y='player_contribution',
    color='efficiency',
    size='ppr_sh',
    hover_data=['player_name','team','position','season'],
    color_continuous_scale='Viridis',
    title='Player Contribution vs Snap %'
)

# Save plots to HTML
pio.write_html(fig1, file="nfl_airyards_plot.html", auto_open=False)
pio.write_html(fig2, file="nfl_contrib_plot.html", auto_open=False)

# === 6. Generate Dashboard HTML ===
html_content = f"""
<html>
<head>
<title>NFL Super Dashboard (2022-2024)</title>
</head>
<body>
<h1>NFL Super Analytics Dashboard</h1>
<p>Report generated: {datetime.now()}</p>

<h2>Top 10 Players by Contribution Index</h2>
{top10_contrib[['player_name','team','position','season','player_contribution','efficiency','snap_pct']].to_html(index=False)}

<h2>Air Yards vs YAC Impact</h2>
<iframe src="nfl_airyards_plot.html" width="100%" height="600"></iframe>

<h2>Player Contribution vs Snap %</h2>
<iframe src="nfl_contrib_plot.html" width="100%" height="600"></iframe>

</body>
</html>
"""

with open("nfl_super_dashboard.html", "w") as f:
    f.write(html_content)

print("NFL Super Dashboard generated: nfl_super_dashboard.html")