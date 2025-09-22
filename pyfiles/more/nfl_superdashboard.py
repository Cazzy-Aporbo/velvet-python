import nfl_data_py as nfl
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime

# === 1. Seasons to include ===
years = [2022, 2023, 2024]

# === 2. Pull Data ===
print("Fetching data...")
pbp = nfl.import_pbp_data(years=years, downcast=True)
weekly = nfl.import_weekly_data(years=years)
seasonal = nfl.import_seasonal_data(years=years, s_type='REG')
rosters = nfl.import_seasonal_rosters(years=years)
snaps = nfl.import_snap_counts(years=years)

# === 3. Clean data ===
for df in [pbp, weekly, seasonal, rosters]:
    try:
        df = nfl.clean_nfl_data(df)
    except Exception as e:
        print(f"Warning: Could not clean df: {e}")

# === 4. Ensure columns exist ===
required_cols = ['dom', 'wopr', 'ppr_sh', 'ay_sh', 'yac_sh', 'player_id', 'player_name', 'team', 'position', 'season']
for col in required_cols:
    if col not in seasonal.columns:
        seasonal[col] = 0

for col in ['player_id','season','snaps']:
    if col not in snaps.columns:
        snaps[col] = 0

# === 5. Merge datasets ===
df = pd.merge(seasonal, rosters, on=['player_id','season'], how='left')
df = pd.merge(df, snaps, on=['player_id','season'], how='left')
df['snaps'] = df['snaps'].fillna(0)
df['snap_pct'] = df['snaps'] / df['snaps'].max().replace(0,1)  # avoid division by 0

# === 6. Compute Advanced Metrics ===
df['efficiency'] = df['dom'] * df['wopr']
df['air_yards_impact'] = df['ay_sh'] * df['yac_sh']
df['player_contribution'] = (df['ppr_sh'] + df['efficiency']) * df['snap_pct']

# === 7. Top Performers ===
top10_contrib = df.sort_values('player_contribution', ascending=False).head(10)

# === 8. Interactive Plots ===
fig1 = px.scatter(
    df, x='ay_sh', y='yac_sh',
    color='air_yards_impact',
    size='player_contribution',
    hover_data=['player_name','team','position','season'],
    color_continuous_scale='Turbo',
    title='Air Yards vs YAC Impact'
)

fig2 = px.scatter(
    df, x='snap_pct', y='player_contribution',
    color='efficiency',
    size='ppr_sh',
    hover_data=['player_name','team','position','season'],
    color_continuous_scale='Viridis',
    title='Player Contribution vs Snap %'
)

# Save plots
pio.write_html(fig1, file="nfl_airyards_plot.html", auto_open=False)
pio.write_html(fig2, file="nfl_contrib_plot.html", auto_open=False)

# === 9. Generate Dashboard HTML ===
html_content = f"""
<html>
<head><title>NFL Super Dashboard (Robust)</title></head>
<body>
<h1>NFL Super Analytics Dashboard (2022-2024)</h1>
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

with open("nfl_super_dashboard_robust.html", "w") as f:
    f.write(html_content)

print("✅ NFL Super Dashboard (robust) generated: nfl_super_dashboard_robust.html")