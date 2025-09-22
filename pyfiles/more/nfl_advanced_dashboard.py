import nfl_data_py as nfl
import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import datetime

# === 1. Import Multiple NFL Datasets ===

years = [2022, 2023, 2024]

# Play-by-play
pbp = nfl.import_pbp_data(years=years, downcast=True)

# Weekly team stats
weekly = nfl.import_weekly_data(years=years)

# Seasonal player stats
seasonal = nfl.import_seasonal_data(years=years, s_type='REG')

# Roster info
rosters = nfl.import_seasonal_rosters(years=years)

# Draft picks
draft = nfl.import_draft_picks(years=years)

# Combine data
combine = nfl.import_combine_data(years=years, positions=['QB','WR','RB','TE'])

# Injuries
injuries = nfl.import_injuries(years=years)

# Snap counts
snaps = nfl.import_snap_counts(years=years)

# Clean key datasets
pbp = nfl.clean_nfl_data(pbp)
weekly = nfl.clean_nfl_data(weekly)
seasonal = nfl.clean_nfl_data(seasonal)
rosters = nfl.clean_nfl_data(rosters)

# === 2. Merge Data for Advanced Analysis ===
# Example: Merge seasonal stats with roster info
df = pd.merge(seasonal, rosters, on=['player_id','season'], how='left')

# Example: Compute "Efficiency Score"
df['efficiency'] = df['dom'].fillna(0) * df['wopr'].fillna(0)  # Dominator * Weighted Opportunity

# === 3. Top Performers by Efficiency ===
top10_eff = df.sort_values('efficiency', ascending=False).head(10)

# === 4. Interactive Color-Coded Plots ===
fig_eff = px.scatter(
    df,
    x='ay_sh',           # Air yards share
    y='yac_sh',          # Yards after catch share
    color='efficiency',
    size='ppr_sh',
    hover_data=['player_name','team','position','season'],
    color_continuous_scale='Viridis',
    title='NFL Player Efficiency vs Air Yards / YAC (2022-2024)'
)

# Save plot to HTML
pio.write_html(fig_eff, file="nfl_efficiency_plot.html", auto_open=False)

# === 5. Generate Dashboard HTML ===
html_content = f"""
<html>
<head><title>NFL Advanced Dashboard</title></head>
<body>
<h1>NFL Advanced Analytics Dashboard (2022-2024)</h1>
<p>Report generated: {datetime.now()}</p>

<h2>Top 10 Players by Efficiency Score</h2>
{top10_eff[['player_name','team','position','season','efficiency','dom','wopr','ppr_sh']].to_html(index=False)}

<h2>Efficiency vs Air Yards / YAC</h2>
<iframe src="nfl_efficiency_plot.html" width="100%" height="600"></iframe>

</body>
</html>
"""

with open("nfl_advanced_dashboard.html", "w") as f:
    f.write(html_content)

print("NFL Advanced Dashboard generated: nfl_advanced_dashboard.html")