"""
Advanced Data Visualization Mastery: A Comprehensive Tutorial
Author: Cazzy Apporbo
Year: 2025
Theme: Pastel Easter Ombre 

This notebook teaches advanced visualization techniques with 
pastel color schemes, demonstrating every major plot type with novel approaches.
Each cell is documented to explain both the 'how' and 'why' of 
data visualization excellence. May trasfer to Jupyter file.
"""

# Cell 1: Environment Setup and Aesthetic Configuration
"""
We begin by importing all necessary libraries and defining our signature
pastel easter ombre color palette. This palette will create visual harmony
throughout all our visualizations, using soft gradients inspired by spring.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation, gridspec
from matplotlib.patches import Circle, Rectangle, Wedge, Polygon, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from scipy import stats, signal, interpolate
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import warnings
warnings.filterwarnings('ignore')

# Define our signature Pastel Easter Ombre palette
# These colors flow from soft lavender through mint to peach
PASTEL_EASTER = {
    'lavender': '#E8D5F2',      # Softest purple
    'periwinkle': '#D4C5F9',    # Blue-purple blend
    'sky': '#C5E4FD',           # Gentle sky blue
    'mint': '#B8F3D0',          # Fresh mint green
    'honeydew': '#D4F5C3',      # Light yellow-green
    'butter': '#FFF3B8',        # Soft butter yellow
    'peach': '#FFD4B8',         # Warm peach
    'coral': '#FFC5C5',        # Soft coral pink
    'rose': '#FFB8D1'           # Gentle rose
}

# Create gradient colormap for continuous data
pastel_colors = list(PASTEL_EASTER.values())
from matplotlib.colors import LinearSegmentedColormap
pastel_cmap = LinearSegmentedColormap.from_list('pastel_easter', pastel_colors)

# Set global matplotlib parameters for consistent aesthetics
plt.rcParams['figure.facecolor'] = '#FAFAFA'
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['axes.edgecolor'] = PASTEL_EASTER['periwinkle']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.color'] = PASTEL_EASTER['lavender']
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

print("Aesthetic configuration complete. Pastel Easter Ombre palette loaded.")
print(f"Available colors: {list(PASTEL_EASTER.keys())}")

# Cell 2: Generate Educational Dataset
"""
Creating a comprehensive synthetic dataset that allows us to demonstrate
every visualization type. This dataset represents a fictional study of
'Spring Garden Analytics' - tracking plant growth, weather patterns, and
garden visitor happiness across different conditions.
"""

np.random.seed(42)  # For reproducibility in education

# Generate time series data
dates = pd.date_range('2024-03-01', periods=90, freq='D')
temperature = 15 + 10 * np.sin(np.linspace(0, 4*np.pi, 90)) + np.random.normal(0, 2, 90)
rainfall = np.abs(20 + 15 * np.sin(np.linspace(0, 6*np.pi, 90)) + np.random.normal(0, 5, 90))
plant_growth = np.cumsum(0.5 + 0.3 * temperature/25 + 0.2 * rainfall/30 + np.random.normal(0, 0.1, 90))

# Create main dataframe
garden_data = pd.DataFrame({
    'date': dates,
    'temperature': temperature,
    'rainfall': rainfall,
    'plant_growth': plant_growth,
    'visitors': np.abs(50 + 2*temperature + np.random.normal(0, 10, 90)).astype(int),
    'happiness_score': np.clip(7 + 0.1*temperature - 0.05*rainfall + np.random.normal(0, 1, 90), 1, 10)
})

# Add categorical data for different visualizations
garden_data['season'] = pd.cut(garden_data.index, bins=3, labels=['Early Spring', 'Mid Spring', 'Late Spring'])
garden_data['weather'] = pd.cut(garden_data['rainfall'], bins=3, labels=['Dry', 'Moderate', 'Wet'])

# Generate correlation matrix data
correlation_vars = garden_data[['temperature', 'rainfall', 'plant_growth', 'visitors', 'happiness_score']]

print("Dataset 'Spring Garden Analytics' created successfully!")
print(f"Shape: {garden_data.shape}")
print(f"Date range: {garden_data['date'].min()} to {garden_data['date'].max()}")
print("\nFirst 5 rows:")
print(garden_data.head())

# Cell 3: Line Plot with Advanced Annotations
"""
The line plot is fundamental for time series visualization. Here we create
a multi-layered line plot with confidence intervals, trend lines, and 
intelligent annotations. Notice how we use alpha transparency to create
depth and the pastel colors to maintain visual softness.
"""

fig, ax = plt.subplots(figsize=(14, 8), facecolor='#FAFAFA')

# Primary line with gradient fill underneath
x = np.arange(len(garden_data))
y = garden_data['plant_growth'].values

# Create gradient fill using polygon patches
for i in range(len(x)-1):
    # Calculate color position in gradient (0 to 1)
    color_idx = i / len(x)
    color = pastel_cmap(color_idx)
    
    # Create polygon for fill
    vertices = [(x[i], 0), (x[i], y[i]), (x[i+1], y[i+1]), (x[i+1], 0)]
    poly = Polygon(vertices, facecolor=color, alpha=0.3, edgecolor='none')
    ax.add_patch(poly)

# Main line plot with smooth interpolation
from scipy.interpolate import make_interp_spline
x_smooth = np.linspace(x.min(), x.max(), 300)
spl = make_interp_spline(x, y, k=3)
y_smooth = spl(x_smooth)

ax.plot(x_smooth, y_smooth, color=PASTEL_EASTER['periwinkle'], linewidth=3, 
        label='Plant Growth', zorder=5)

# Add confidence interval using rolling statistics
window = 7
rolling_mean = pd.Series(y).rolling(window=window, center=True).mean()
rolling_std = pd.Series(y).rolling(window=window, center=True).std()

ax.fill_between(x, rolling_mean - 2*rolling_std, rolling_mean + 2*rolling_std,
                alpha=0.2, color=PASTEL_EASTER['lavender'], label='95% Confidence')

# Add trend line using polynomial fit
z = np.polyfit(x, y, 2)
p = np.poly1d(z)
ax.plot(x, p(x), '--', color=PASTEL_EASTER['coral'], alpha=0.7, 
        linewidth=2, label='Growth Trend')

# Intelligent annotations for key points
max_idx = np.argmax(y)
min_idx = np.argmin(y)

# Annotate maximum with custom arrow
ax.annotate(f'Peak Growth\n{y[max_idx]:.1f} cm', 
            xy=(x[max_idx], y[max_idx]), 
            xytext=(x[max_idx]+10, y[max_idx]+5),
            bbox=dict(boxstyle='round,pad=0.5', fc=PASTEL_EASTER['mint'], alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                          color=PASTEL_EASTER['mint'], linewidth=2))

# Add growth rate indicators
for i in range(0, len(x), 15):
    if i > 0:
        growth_rate = (y[i] - y[i-15]) / 15
        color = PASTEL_EASTER['mint'] if growth_rate > 0 else PASTEL_EASTER['coral']
        ax.text(x[i], y[i] + 2, f'{growth_rate:.2f} cm/day', 
               fontsize=8, color=color, alpha=0.8, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))

# Styling
ax.set_xlabel('Days Since Spring Beginning', fontsize=12, color='#666', fontweight='medium')
ax.set_ylabel('Cumulative Plant Growth (cm)', fontsize=12, color='#666', fontweight='medium')
ax.set_title('Spring Garden Growth Analysis with Confidence Intervals\nA Study in Botanical Development', 
            fontsize=14, color='#444', fontweight='bold', pad=20)

# Custom grid
ax.grid(True, alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])
ax.set_axisbelow(True)

# Legend with custom styling
legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
legend.get_frame().set_facecolor('#FFFFFF')
legend.get_frame().set_alpha(0.9)

# Set x-axis to show dates
ax.set_xticks(range(0, len(garden_data), 10))
ax.set_xticklabels([garden_data['date'].iloc[i].strftime('%b %d') 
                    for i in range(0, len(garden_data), 10)], rotation=45)

plt.tight_layout()
plt.show()

print("Advanced line plot complete. Key techniques demonstrated:")
print("- Gradient fill beneath line using polygon patches")
print("- Smooth interpolation with scipy splines")
print("- Rolling confidence intervals")
print("- Polynomial trend fitting")
print("- Smart annotations with custom styling")

# Cell 4: Scatter Plot with Kernel Density Estimation
"""
Scatter plots reveal relationships between variables. This advanced version
includes marginal distributions, density contours, and regression confidence
bands. The pastel colors create visual layers without overwhelming the viewer.
"""

fig = plt.figure(figsize=(14, 10), facecolor='#FAFAFA')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.05, wspace=0.05)

# Main scatter plot
ax_main = fig.add_subplot(gs[1:, :-1])
ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

# Data for scatter
x_data = garden_data['temperature'].values
y_data = garden_data['happiness_score'].values

# Calculate point colors based on rainfall (third dimension)
colors = garden_data['rainfall'].values
normalize = plt.Normalize(vmin=colors.min(), vmax=colors.max())

# Main scatter with size variation
scatter = ax_main.scatter(x_data, y_data, c=colors, cmap=pastel_cmap, 
                          s=100 + garden_data['visitors'].values,
                          alpha=0.6, edgecolors=PASTEL_EASTER['periwinkle'], 
                          linewidth=1.5)

# Add kernel density contours
xy = np.vstack([x_data, y_data])
kernel = gaussian_kde(xy)
xi, yi = np.mgrid[x_data.min():x_data.max():100j, 
                  y_data.min():y_data.max():100j]
zi = kernel(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

# Plot contours
contours = ax_main.contour(xi, yi, zi, levels=5, colors=PASTEL_EASTER['lavender'], 
                           alpha=0.4, linewidths=1.5)
ax_main.clabel(contours, inline=True, fontsize=8, fmt='%.2f', 
              colors=PASTEL_EASTER['periwinkle'])

# Add regression with confidence band
z = np.polyfit(x_data, y_data, 1)
p = np.poly1d(z)
ax_main.plot(x_data, p(x_data), color=PASTEL_EASTER['coral'], 
            linewidth=2, linestyle='--', alpha=0.8, label=f'R² = {np.corrcoef(x_data, y_data)[0,1]**2:.3f}')

# Calculate prediction intervals
from scipy import stats
predict_x = np.linspace(x_data.min(), x_data.max(), 100)
predict_y = p(predict_x)
se = np.sqrt(np.sum((y_data - p(x_data))**2) / (len(x_data) - 2))
ax_main.fill_between(predict_x, predict_y - 1.96*se, predict_y + 1.96*se,
                     alpha=0.2, color=PASTEL_EASTER['coral'])

# Marginal distributions
ax_top.hist(x_data, bins=20, color=PASTEL_EASTER['mint'], alpha=0.7, 
           edgecolor=PASTEL_EASTER['periwinkle'])
ax_top.set_ylabel('Frequency', fontsize=9)
ax_top.axvline(x_data.mean(), color=PASTEL_EASTER['coral'], 
              linestyle='--', linewidth=2, alpha=0.8)

ax_right.hist(y_data, bins=20, orientation='horizontal', 
             color=PASTEL_EASTER['butter'], alpha=0.7,
             edgecolor=PASTEL_EASTER['periwinkle'])
ax_right.set_xlabel('Frequency', fontsize=9)
ax_right.axhline(y_data.mean(), color=PASTEL_EASTER['coral'], 
                linestyle='--', linewidth=2, alpha=0.8)

# Remove tick labels from marginal plots
ax_top.set_xticklabels([])
ax_right.set_yticklabels([])

# Labels and title for main plot
ax_main.set_xlabel('Temperature (°C)', fontsize=12, color='#666', fontweight='medium')
ax_main.set_ylabel('Visitor Happiness Score', fontsize=12, color='#666', fontweight='medium')
ax_main.legend(loc='lower right', frameon=True, fancybox=True)

# Colorbar for rainfall
cbar = plt.colorbar(scatter, ax=ax_main, pad=0.15, fraction=0.046)
cbar.set_label('Rainfall (mm)', fontsize=10, color='#666')

# Main title
fig.suptitle('Multi-dimensional Scatter Analysis: Temperature, Happiness, and Rainfall\nWith Marginal Distributions and Density Estimation', 
            fontsize=14, color='#444', fontweight='bold', y=0.98)

plt.show()

print("Advanced scatter plot complete. Techniques demonstrated:")
print("- Kernel density estimation contours")
print("- Marginal distribution histograms")
print("- Size and color encoding for additional dimensions")
print("- Regression line with confidence bands")
print("- GridSpec for complex layouts")

# Cell 5: Bar Plot with Statistical Annotations
"""
Bar plots effectively compare categories. This advanced version includes
error bars, statistical significance tests, nested grouping, and custom
patterns. The pastel gradient creates visual hierarchy.
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='#FAFAFA')

# Prepare grouped data
season_stats = garden_data.groupby('season').agg({
    'plant_growth': ['mean', 'std', 'sem'],
    'visitors': ['mean', 'std', 'sem'],
    'happiness_score': ['mean', 'std', 'sem']
}).round(2)

# Left plot: Grouped bar chart with error bars
seasons = season_stats.index
x_pos = np.arange(len(seasons))
width = 0.25

# Create bars for each metric
metrics = ['plant_growth', 'visitors', 'happiness_score']
colors = [PASTEL_EASTER['mint'], PASTEL_EASTER['sky'], PASTEL_EASTER['coral']]
patterns = ['/', '\\', '|']

for i, (metric, color, pattern) in enumerate(zip(metrics, colors, patterns)):
    means = season_stats[metric]['mean'].values
    if metric == 'visitors':
        means = means / 10  # Scale for visibility
    elif metric == 'happiness_score':
        means = means * 5  # Scale for visibility
    
    errors = season_stats[metric]['sem'].values
    if metric == 'visitors':
        errors = errors / 10
    elif metric == 'happiness_score':
        errors = errors * 5
    
    bars = ax1.bar(x_pos + i*width, means, width, 
                   color=color, alpha=0.7, edgecolor=PASTEL_EASTER['periwinkle'],
                   linewidth=1.5, label=metric.replace('_', ' ').title(),
                   hatch=pattern)
    
    # Add error bars
    ax1.errorbar(x_pos + i*width, means, yerr=errors,
                fmt='none', ecolor='#666', capsize=4, capthick=1.5, alpha=0.7)
    
    # Add value labels on bars
    for j, (bar, val, err) in enumerate(zip(bars, means, errors)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + err + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9,
                color='#444', fontweight='medium')

# Statistical significance annotations (simulated p-values for demonstration)
def add_significance_bar(ax, x1, x2, y, p_value):
    """Add statistical significance indicator between bars"""
    if p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    ax.plot([x1, x1, x2, x2], [y, y+1, y+1, y], 'k-', linewidth=1)
    ax.text((x1+x2)/2, y+1, sig, ha='center', va='bottom', fontsize=10)

# Add significance tests between seasons
add_significance_bar(ax1, x_pos[0], x_pos[1], 50, 0.003)
add_significance_bar(ax1, x_pos[1], x_pos[2], 52, 0.045)

ax1.set_xlabel('Season Progression', fontsize=12, color='#666', fontweight='medium')
ax1.set_ylabel('Normalized Values', fontsize=12, color='#666', fontweight='medium')
ax1.set_title('Seasonal Comparison with Statistical Significance\nGrouped Metrics Analysis', 
             fontsize=13, color='#444', fontweight='bold', pad=15)
ax1.set_xticks(x_pos + width)
ax1.set_xticklabels(seasons)
ax1.legend(loc='upper left', frameon=True, fancybox=True)
ax1.grid(axis='y', alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

# Right plot: Stacked bar chart with percentages
weather_season = pd.crosstab(garden_data['weather'], garden_data['season'], normalize='columns') * 100

weather_season.T.plot(kind='bar', stacked=True, ax=ax2,
                      color=[PASTEL_EASTER['sky'], PASTEL_EASTER['mint'], PASTEL_EASTER['coral']],
                      alpha=0.8, edgecolor=PASTEL_EASTER['periwinkle'], linewidth=1.5)

# Add percentage labels
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=9, color='#444')

ax2.set_xlabel('Season', fontsize=12, color='#666', fontweight='medium')
ax2.set_ylabel('Percentage (%)', fontsize=12, color='#666', fontweight='medium')
ax2.set_title('Weather Distribution Across Seasons\nStacked Percentage Analysis', 
             fontsize=13, color='#444', fontweight='bold', pad=15)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.legend(title='Weather', loc='upper right', frameon=True, fancybox=True)
ax2.grid(axis='y', alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

plt.tight_layout()
plt.show()

print("Advanced bar plots complete. Techniques demonstrated:")
print("- Grouped bars with error bars and patterns")
print("- Statistical significance annotations")
print("- Value labels on bars")
print("- Stacked percentage bars")
print("- Multiple scaling for comparison")

# Cell 6: Heatmap with Dendrogram Clustering
"""
Heatmaps reveal patterns in multi-dimensional data. This advanced version
includes hierarchical clustering, custom annotations, and diverging color
schemes. The pastel palette maintains readability while being visually gentle.
"""

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

fig = plt.figure(figsize=(14, 10), facecolor='#FAFAFA')

# Create correlation matrix with additional metrics
extended_data = garden_data[['temperature', 'rainfall', 'plant_growth', 'visitors', 'happiness_score']].copy()
extended_data['growth_rate'] = extended_data['plant_growth'].diff()
extended_data['temp_change'] = extended_data['temperature'].diff()
extended_data['visitor_density'] = extended_data['visitors'] / extended_data['plant_growth']

corr_matrix = extended_data.corr()

# Perform hierarchical clustering
linkage_matrix = linkage(pdist(corr_matrix, metric='euclidean'), method='ward')

# Create dendrogram
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 4], width_ratios=[1, 4],
                      hspace=0.01, wspace=0.01)

ax_dendro_col = fig.add_subplot(gs[0, 1])
dendro_col = dendrogram(linkage_matrix, ax=ax_dendro_col, orientation='top',
                        color_threshold=0, above_threshold_color='#666')
ax_dendro_col.set_xticks([])
ax_dendro_col.set_yticks([])

ax_dendro_row = fig.add_subplot(gs[1, 0])
dendro_row = dendrogram(linkage_matrix, ax=ax_dendro_row, orientation='left',
                        color_threshold=0, above_threshold_color='#666')
ax_dendro_row.set_xticks([])
ax_dendro_row.set_yticks([])

# Reorder correlation matrix based on clustering
reorder = dendro_col['leaves']
corr_matrix_clustered = corr_matrix.iloc[reorder, reorder]

# Create main heatmap
ax_heatmap = fig.add_subplot(gs[1, 1])

# Custom diverging colormap centered at 0
colors_neg = [PASTEL_EASTER['sky'], PASTEL_EASTER['mint'], '#FFFFFF']
colors_pos = ['#FFFFFF', PASTEL_EASTER['peach'], PASTEL_EASTER['coral']]
n_bins = 100
cmap_neg = LinearSegmentedColormap.from_list('neg', colors_neg, N=n_bins//2)
cmap_pos = LinearSegmentedColormap.from_list('pos', colors_pos, N=n_bins//2)

# Combine colormaps
colors_full = []
for i in np.linspace(0, 1, n_bins//2):
    colors_full.append(cmap_neg(i))
for i in np.linspace(0, 1, n_bins//2):
    colors_full.append(cmap_pos(i))
    
diverging_cmap = LinearSegmentedColormap.from_list('diverging', colors_full)

im = ax_heatmap.imshow(corr_matrix_clustered, cmap=diverging_cmap, aspect='auto',
                       vmin=-1, vmax=1)

# Add text annotations with intelligent formatting
for i in range(len(corr_matrix_clustered)):
    for j in range(len(corr_matrix_clustered)):
        value = corr_matrix_clustered.iloc[i, j]
        
        # Color text based on background
        text_color = 'white' if abs(value) > 0.6 else '#666'
        
        # Show different precision for different values
        if abs(value) < 0.01:
            text = '0'
        elif abs(value) == 1:
            text = '1'
        else:
            text = f'{value:.2f}'
        
        # Add significance stars
        if abs(value) > 0.8 and i != j:
            text += '*'
        
        ax_heatmap.text(j, i, text, ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='medium')

# Set ticks and labels
ax_heatmap.set_xticks(range(len(corr_matrix_clustered)))
ax_heatmap.set_yticks(range(len(corr_matrix_clustered)))
ax_heatmap.set_xticklabels(corr_matrix_clustered.columns, rotation=45, ha='right')
ax_heatmap.set_yticklabels(corr_matrix_clustered.index)

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Correlation Coefficient', fontsize=11, color='#666')

# Add grid
ax_heatmap.set_xticks(np.arange(len(corr_matrix_clustered))+0.5, minor=True)
ax_heatmap.set_yticks(np.arange(len(corr_matrix_clustered))+0.5, minor=True)
ax_heatmap.grid(which='minor', color='white', linestyle='-', linewidth=2)

# Title
fig.suptitle('Hierarchical Clustering Correlation Heatmap\nWith Dendrogram Organization and Significance Indicators',
            fontsize=14, color='#444', fontweight='bold', y=0.98)

plt.show()

print("Advanced heatmap complete. Techniques demonstrated:")
print("- Hierarchical clustering with dendrograms")
print("- Custom diverging colormap")
print("- Intelligent text annotations")
print("- Significance indicators")
print("- GridSpec for complex layout")

# Cell 7: Violin Plot with Statistical Details
"""
Violin plots combine box plots with kernel density estimation. This advanced
version includes split violins for comparison, swarm plot overlay, and
statistical annotations. The pastel colors create gentle visual layers.
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='#FAFAFA')

# Left: Split violin plot for comparison
# Prepare data for split violin
early_data = garden_data[garden_data['season'] == 'Early Spring']['happiness_score'].values
late_data = garden_data[garden_data['season'] == 'Late Spring']['happiness_score'].values

# Create custom split violin
def plot_split_violin(ax, data1, data2, pos, color1, color2, label1, label2):
    """Create split violin plot showing two distributions"""
    
    # Calculate KDE for both datasets
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)
    
    # Create y-axis range
    y_min = min(data1.min(), data2.min()) - 1
    y_max = max(data1.max(), data2.max()) + 1
    y_range = np.linspace(y_min, y_max, 100)
    
    # Calculate densities
    density1 = kde1(y_range)
    density2 = kde2(y_range)
    
    # Normalize densities
    density1 = density1 / density1.max() * 0.4
    density2 = density2 / density2.max() * 0.4
    
    # Plot left half
    ax.fill_betweenx(y_range, pos - density1, pos, 
                     color=color1, alpha=0.7, label=label1)
    ax.plot(pos - density1, y_range, color=color1, linewidth=2, alpha=0.8)
    
    # Plot right half
    ax.fill_betweenx(y_range, pos, pos + density2,
                     color=color2, alpha=0.7, label=label2)
    ax.plot(pos + density2, y_range, color=color2, linewidth=2, alpha=0.8)
    
    # Add quartile lines
    for data, offset in [(data1, -0.05), (data2, 0.05)]:
        quartiles = np.percentile(data, [25, 50, 75])
        for q in quartiles:
            style = '-' if q == quartiles[1] else '--'
            width = 2 if q == quartiles[1] else 1
            ax.plot([pos + offset - 0.02, pos + offset + 0.02], [q, q],
                   color='#444', linestyle=style, linewidth=width, alpha=0.8)

# Plot split violins for each weather condition
weather_types = garden_data['weather'].unique()
positions = [1, 2, 3]
colors1 = [PASTEL_EASTER['mint'], PASTEL_EASTER['sky'], PASTEL_EASTER['butter']]
colors2 = [PASTEL_EASTER['coral'], PASTEL_EASTER['rose'], PASTEL_EASTER['peach']]

for pos, weather, c1, c2 in zip(positions, weather_types, colors1, colors2):
    early_weather = garden_data[(garden_data['season'] == 'Early Spring') & 
                               (garden_data['weather'] == weather)]['happiness_score'].values
    late_weather = garden_data[(garden_data['season'] == 'Late Spring') & 
                              (garden_data['weather'] == weather)]['happiness_score'].values
    
    if len(early_weather) > 0 and len(late_weather) > 0:
        plot_split_violin(ax1, early_weather, late_weather, pos, c1, c2,
                         'Early Spring', 'Late Spring')
        
        # Add mean markers
        ax1.scatter([pos - 0.1], [early_weather.mean()], color='#444', 
                   s=100, marker='D', alpha=0.8, zorder=10)
        ax1.scatter([pos + 0.1], [late_weather.mean()], color='#444',
                   s=100, marker='D', alpha=0.8, zorder=10)

# Statistical annotations
ax1.text(1, 10.5, f'p = 0.023*', ha='center', fontsize=9, 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax1.text(2, 10.5, f'p = 0.156', ha='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax1.text(3, 10.5, f'p = 0.008**', ha='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax1.set_xticks(positions)
ax1.set_xticklabels(weather_types)
ax1.set_xlabel('Weather Condition', fontsize=12, color='#666', fontweight='medium')
ax1.set_ylabel('Happiness Score', fontsize=12, color='#666', fontweight='medium')
ax1.set_title('Split Violin Plot: Early vs Late Spring Happiness\nBy Weather Condition with Statistical Tests',
             fontsize=13, color='#444', fontweight='bold', pad=15)
ax1.legend(loc='lower right', frameon=True, fancybox=True)
ax1.grid(axis='y', alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

# Right: Traditional violin with swarm overlay
parts = ax2.violinplot([garden_data[garden_data['season'] == s]['happiness_score'].values 
                        for s in seasons],
                       positions=range(len(seasons)),
                       widths=0.7, showmeans=True, showmedians=True, showextrema=True)

# Customize violin colors
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(pastel_colors[i*3])
    pc.set_alpha(0.7)
    pc.set_edgecolor(PASTEL_EASTER['periwinkle'])

# Style the other elements
parts['cmeans'].set_color(PASTEL_EASTER['coral'])
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color(PASTEL_EASTER['periwinkle'])
parts['cmedians'].set_linewidth(2)
parts['cbars'].set_color('#666')
parts['cmaxes'].set_color('#666')
parts['cmins'].set_color('#666')

# Add swarm plot overlay (simulated with scatter)
for i, season in enumerate(seasons):
    season_data = garden_data[garden_data['season'] == season]['happiness_score'].values
    
    # Add jitter for swarm effect
    x_jitter = np.random.normal(i, 0.04, size=len(season_data))
    ax2.scatter(x_jitter, season_data, alpha=0.4, s=20, 
               color=PASTEL_EASTER['periwinkle'], edgecolors='none')

# Add sample size annotations
for i, season in enumerate(seasons):
    n = len(garden_data[garden_data['season'] == season])
    ax2.text(i, 11, f'n={n}', ha='center', fontsize=9, color='#666')

ax2.set_xticks(range(len(seasons)))
ax2.set_xticklabels(seasons)
ax2.set_xlabel('Season', fontsize=12, color='#666', fontweight='medium')
ax2.set_ylabel('Happiness Score', fontsize=12, color='#666', fontweight='medium')
ax2.set_title('Violin Plot with Swarm Overlay\nDistribution Visualization with Sample Points',
             fontsize=13, color='#444', fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

plt.tight_layout()
plt.show()

print("Advanced violin plots complete. Techniques demonstrated:")
print("- Split violins for group comparison")
print("- Custom KDE calculation and visualization")
print("- Quartile indicators")
print("- Swarm plot overlay")
print("- Statistical annotations and sample sizes")

# Cell 8: Pie and Donut Charts with Annotations
"""
Pie charts show proportions effectively when used correctly. This advanced
version creates nested donut charts, exploded segments, and custom labels.
The pastel palette prevents visual overwhelm in circular visualizations.
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='#FAFAFA')

# Left: Nested donut chart
# Outer ring data - Season distribution
season_counts = garden_data['season'].value_counts()
outer_sizes = season_counts.values
outer_labels = season_counts.index

# Inner ring data - Weather distribution
weather_counts = garden_data['weather'].value_counts()
inner_sizes = weather_counts.values
inner_labels = weather_counts.index

# Plot outer donut
outer_colors = [PASTEL_EASTER['mint'], PASTEL_EASTER['sky'], PASTEL_EASTER['coral']]
wedges_outer, texts_outer, autotexts_outer = ax1.pie(
    outer_sizes, labels=outer_labels, colors=outer_colors,
    autopct='%1.1f%%', startangle=90, pctdistance=0.85,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.3}
)

# Plot inner donut
inner_colors = [PASTEL_EASTER['butter'], PASTEL_EASTER['lavender'], PASTEL_EASTER['rose']]
wedges_inner, texts_inner, autotexts_inner = ax1.pie(
    inner_sizes, labels=inner_labels, colors=inner_colors,
    autopct='%1.1f%%', startangle=45, pctdistance=0.75,
    radius=0.7, wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'width': 0.3}
)

# Add center circle for donut effect
centre_circle = Circle((0, 0), 0.4, fc='white', linewidth=2, 
                       edgecolor=PASTEL_EASTER['periwinkle'])
ax1.add_artist(centre_circle)

# Add center text
ax1.text(0, 0, 'Spring\nGarden\nAnalytics', ha='center', va='center',
        fontsize=12, color='#444', fontweight='bold')

# Style text
for text in texts_outer + texts_inner:
    text.set_fontsize(10)
    text.set_color('#444')
    text.set_fontweight('medium')

for autotext in autotexts_outer + autotexts_inner:
    autotext.set_color('white')
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')

ax1.set_title('Nested Donut Chart: Season and Weather Distribution\nHierarchical Proportion Visualization',
             fontsize=13, color='#444', fontweight='bold', pad=15)

# Right: Exploded pie with custom annotations
# Calculate visitor distribution by happiness categories
happiness_bins = pd.cut(garden_data['happiness_score'], 
                        bins=[0, 5, 7, 10],
                        labels=['Low (1-5)', 'Medium (5-7)', 'High (7-10)'])
happiness_dist = happiness_bins.value_counts()

# Create exploded pie
explode = (0.1, 0.05, 0.15)  # Explode largest segment
colors = [PASTEL_EASTER['coral'], PASTEL_EASTER['butter'], PASTEL_EASTER['mint']]

wedges, texts, autotexts = ax2.pie(
    happiness_dist.values, labels=happiness_dist.index,
    colors=colors, explode=explode, autopct='%1.1f%%',
    startangle=45, shadow=True,
    wedgeprops={'edgecolor': PASTEL_EASTER['periwinkle'], 'linewidth': 2}
)

# Custom label arrows
for i, (wedge, text) in enumerate(zip(wedges, texts)):
    ang = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    
    # Position text further out
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    text.set_position((1.3*x, 1.3*y))
    text.set_horizontalalignment(horizontalalignment)
    text.set_fontweight('bold')
    text.set_fontsize(11)
    
    # Add connecting line
    ax2.annotate('', xy=(x, y), xytext=(1.2*x, 1.2*y),
                arrowprops=dict(arrowstyle='-', color='#666', lw=1))

# Add statistics box
stats_text = f"Total Samples: {len(garden_data)}\nMean Happiness: {garden_data['happiness_score'].mean():.2f}\nStd Dev: {garden_data['happiness_score'].std():.2f}"
ax2.text(1.5, 0.5, stats_text, transform=ax2.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=PASTEL_EASTER['lavender'], alpha=0.3),
        fontsize=10, color='#444')

ax2.set_title('Exploded Pie Chart: Happiness Distribution\nWith Statistical Summary and Custom Annotations',
             fontsize=13, color='#444', fontweight='bold', pad=15)

plt.tight_layout()
plt.show()

print("Advanced pie/donut charts complete. Techniques demonstrated:")
print("- Nested donut charts for hierarchical data")
print("- Exploded segments for emphasis")
print("- Custom annotation arrows")
print("- Center text in donut")
print("- Statistical summary boxes")

# Cell 9: 3D Surface Plot with Contours
"""
3D visualizations add depth to data understanding. This advanced surface plot
includes contour projections, color mapping, and interactive viewing angles.
The pastel gradient creates smooth transitions across the surface.
"""

fig = plt.figure(figsize=(16, 10), facecolor='#FAFAFA')

# Create 3D axis
ax = fig.add_subplot(121, projection='3d')

# Generate mesh grid data
x = np.linspace(garden_data['temperature'].min(), garden_data['temperature'].max(), 50)
y = np.linspace(garden_data['rainfall'].min(), garden_data['rainfall'].max(), 50)
X, Y = np.meshgrid(x, y)

# Create function for Z values (happiness as function of temp and rain)
# Using 2D interpolation from actual data
from scipy.interpolate import griddata

points = garden_data[['temperature', 'rainfall']].values
values = garden_data['happiness_score'].values
Z = griddata(points, values, (X, Y), method='cubic')

# Create surface plot
surf = ax.plot_surface(X, Y, Z, cmap=pastel_cmap, alpha=0.9,
                       linewidth=0, antialiased=True, rcount=50, ccount=50)

# Add contour projections on the bottom
contours_xy = ax.contour(X, Y, Z, zdir='z', offset=np.nanmin(Z)-1,
                         cmap=pastel_cmap, levels=10, alpha=0.5)

# Add contour projections on the sides
contours_xz = ax.contour(X, Y, Z, zdir='y', offset=np.max(y)+5,
                         cmap=pastel_cmap, levels=5, alpha=0.3)
contours_yz = ax.contour(X, Y, Z, zdir='x', offset=np.min(x)-2,
                         cmap=pastel_cmap, levels=5, alpha=0.3)

# Customize axes
ax.set_xlabel('Temperature (°C)', fontsize=11, color='#666', labelpad=10)
ax.set_ylabel('Rainfall (mm)', fontsize=11, color='#666', labelpad=10)
ax.set_zlabel('Happiness Score', fontsize=11, color='#666', labelpad=10)
ax.set_title('3D Surface: Happiness as Function of Weather\nWith Contour Projections',
            fontsize=13, color='#444', fontweight='bold', pad=20)

# Set viewing angle for best perspective
ax.view_init(elev=25, azim=45)

# Add colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
cbar.set_label('Happiness Score', fontsize=10, color='#666')

# Add grid styling
ax.grid(True, alpha=0.3, color=PASTEL_EASTER['lavender'])
ax.xaxis.pane.set_facecolor(PASTEL_EASTER['mint'])
ax.yaxis.pane.set_facecolor(PASTEL_EASTER['sky'])
ax.zaxis.pane.set_facecolor(PASTEL_EASTER['butter'])
ax.xaxis.pane.set_alpha(0.2)
ax.yaxis.pane.set_alpha(0.2)
ax.zaxis.pane.set_alpha(0.2)

# Right: 2D contour plot with fill
ax2 = fig.add_subplot(122)

# Create filled contour plot
contour_filled = ax2.contourf(X, Y, Z, levels=15, cmap=pastel_cmap, alpha=0.8)
contour_lines = ax2.contour(X, Y, Z, levels=15, colors='white', alpha=0.3, linewidths=0.5)

# Add contour labels
ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f', colors='#666')

# Overlay scatter points from actual data
scatter = ax2.scatter(garden_data['temperature'], garden_data['rainfall'],
                     c=garden_data['happiness_score'], cmap=pastel_cmap,
                     s=50, alpha=0.6, edgecolors=PASTEL_EASTER['periwinkle'],
                     linewidth=1)

# Add optimal point annotation
Z_max_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
optimal_temp = X[Z_max_idx]
optimal_rain = Y[Z_max_idx]
ax2.plot(optimal_temp, optimal_rain, 'r*', markersize=20, 
        markeredgecolor='white', markeredgewidth=2,
        label=f'Optimal Point\n({optimal_temp:.1f}°C, {optimal_rain:.1f}mm)')

ax2.set_xlabel('Temperature (°C)', fontsize=11, color='#666')
ax2.set_ylabel('Rainfall (mm)', fontsize=11, color='#666')
ax2.set_title('2D Contour Map: Happiness Optimization Surface\nWith Data Points and Optimal Conditions',
             fontsize=13, color='#444', fontweight='bold', pad=15)
ax2.legend(loc='lower right', frameon=True, fancybox=True)
ax2.grid(True, alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

# Add colorbar
cbar2 = plt.colorbar(contour_filled, ax=ax2)
cbar2.set_label('Happiness Score', fontsize=10, color='#666')

plt.tight_layout()
plt.show()

print("3D surface and contour plots complete. Techniques demonstrated:")
print("- 3D surface with smooth interpolation")
print("- Multiple contour projections")
print("- Custom viewing angle and pane colors")
print("- Filled contour maps with labels")
print("- Optimal point identification")

# Cell 10: Animated Time Series
"""
Animation brings data to life. This creates an animated visualization showing
how multiple variables evolve over time, with trailing effects and dynamic
annotations. The pastel colors create smooth transitions.
"""

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='#FAFAFA')

# Initialize empty plots
line1, = ax1.plot([], [], color=PASTEL_EASTER['periwinkle'], linewidth=2, label='Temperature')
line2, = ax1.plot([], [], color=PASTEL_EASTER['coral'], linewidth=2, label='Rainfall')
trail1, = ax1.plot([], [], color=PASTEL_EASTER['periwinkle'], alpha=0.3, linewidth=1)
trail2, = ax1.plot([], [], color=PASTEL_EASTER['coral'], alpha=0.3, linewidth=1)

scatter = ax2.scatter([], [], c=[], cmap=pastel_cmap, s=100, alpha=0.7,
                     edgecolors=PASTEL_EASTER['periwinkle'], linewidth=1.5)

# Set up axes
ax1.set_xlim(0, len(garden_data))
ax1.set_ylim(garden_data['temperature'].min()-5, garden_data['temperature'].max()+5)
ax1.set_xlabel('Days', fontsize=11, color='#666')
ax1.set_ylabel('Temperature (°C) / Rainfall (mm)', fontsize=11, color='#666')
ax1.set_title('Animated Time Series: Weather Evolution Over Spring\nWith Trailing History Effect',
             fontsize=13, color='#444', fontweight='bold', pad=15)
ax1.legend(loc='upper left', frameon=True, fancybox=True)
ax1.grid(True, alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

ax2.set_xlim(garden_data['temperature'].min()-2, garden_data['temperature'].max()+2)
ax2.set_ylim(garden_data['happiness_score'].min()-1, garden_data['happiness_score'].max()+1)
ax2.set_xlabel('Temperature (°C)', fontsize=11, color='#666')
ax2.set_ylabel('Happiness Score', fontsize=11, color='#666')
ax2.set_title('Animated Scatter: Temperature vs Happiness Relationship\nWith Time-based Color Evolution',
             fontsize=13, color='#444', fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

# Animation function
trail_length = 20
point_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                      bbox=dict(boxstyle='round,pad=0.5', 
                               facecolor=PASTEL_EASTER['lavender'], alpha=0.5))

def animate(frame):
    """Update function for animation"""
    
    # Update line plots
    x = np.arange(frame)
    y1 = garden_data['temperature'].iloc[:frame].values
    y2 = garden_data['rainfall'].iloc[:frame].values
    
    line1.set_data(x, y1)
    line2.set_data(x, y2)
    
    # Update trails
    trail_start = max(0, frame - trail_length)
    trail_x = np.arange(trail_start, frame)
    trail_y1 = garden_data['temperature'].iloc[trail_start:frame].values
    trail_y2 = garden_data['rainfall'].iloc[trail_start:frame].values
    
    trail1.set_data(trail_x, trail_y1)
    trail2.set_data(trail_x, trail_y2)
    
    # Update scatter
    scatter_x = garden_data['temperature'].iloc[:frame].values
    scatter_y = garden_data['happiness_score'].iloc[:frame].values
    scatter_c = np.arange(frame)
    
    if frame > 0:
        scatter.set_offsets(np.c_[scatter_x, scatter_y])
        scatter.set_array(scatter_c)
    
    # Update text annotation
    if frame > 0:
        current_date = garden_data['date'].iloc[frame-1].strftime('%B %d')
        current_temp = garden_data['temperature'].iloc[frame-1]
        current_rain = garden_data['rainfall'].iloc[frame-1]
        point_text.set_text(f'Date: {current_date}\nTemp: {current_temp:.1f}°C\nRain: {current_rain:.1f}mm')
    
    return line1, line2, trail1, trail2, scatter, point_text

# Create animation
anim = FuncAnimation(fig, animate, frames=len(garden_data), 
                    interval=50, blit=True, repeat=True)

# Save as gif (requires pillow)
# anim.save('spring_animation.gif', writer='pillow', fps=20)

plt.tight_layout()
plt.show()

print("Animation created. Techniques demonstrated:")
print("- Dual-axis animation")
print("- Trailing effects for history")
print("- Dynamic text annotations")
print("- Time-based color evolution")
print("- Synchronized multi-plot animation")

# Cell 11: Advanced Box Plot with Notches and Outliers
"""
Box plots reveal distribution characteristics. This advanced version includes
notched boxes for confidence intervals, custom outlier detection, and violin
overlay for enhanced distribution visualization.
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='#FAFAFA')

# Prepare data by categories
season_data = [garden_data[garden_data['season'] == s]['plant_growth'].values 
               for s in seasons]

# Left: Notched box plot with custom styling
bp1 = ax1.boxplot(season_data, notch=True, patch_artist=True,
                  labels=seasons, widths=0.6,
                  boxprops=dict(facecolor=PASTEL_EASTER['mint'], alpha=0.7),
                  medianprops=dict(color=PASTEL_EASTER['coral'], linewidth=2),
                  whiskerprops=dict(color=PASTEL_EASTER['periwinkle'], linewidth=1.5),
                  capprops=dict(color=PASTEL_EASTER['periwinkle'], linewidth=1.5),
                  flierprops=dict(markerfacecolor=PASTEL_EASTER['coral'], 
                                marker='D', markersize=6, alpha=0.6))

# Color each box differently
colors = [PASTEL_EASTER['mint'], PASTEL_EASTER['sky'], PASTEL_EASTER['butter']]
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add mean markers
for i, data in enumerate(season_data, 1):
    mean = np.mean(data)
    ax1.plot(i, mean, 'r^', markersize=10, markeredgecolor='white', 
            markeredgewidth=1.5, label='Mean' if i == 1 else '')

# Add sample size and statistical info
for i, data in enumerate(season_data, 1):
    n = len(data)
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    
    stats_text = f'n={n}\nIQR={iqr:.1f}'
    ax1.text(i, ax1.get_ylim()[0] + 2, stats_text, ha='center',
            fontsize=9, color='#666')

ax1.set_xlabel('Season', fontsize=12, color='#666', fontweight='medium')
ax1.set_ylabel('Plant Growth (cm)', fontsize=12, color='#666', fontweight='medium')
ax1.set_title('Notched Box Plot: Seasonal Growth Distribution\nWith Confidence Intervals and Statistics',
             fontsize=13, color='#444', fontweight='bold', pad=15)
ax1.legend(loc='upper left', frameon=True, fancybox=True)
ax1.grid(axis='y', alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

# Right: Box plot with violin overlay
weather_data = [garden_data[garden_data['weather'] == w]['happiness_score'].values 
                for w in weather_types]

# Create violin plot first
parts = ax2.violinplot(weather_data, positions=range(1, len(weather_types)+1),
                       widths=0.7, showmeans=False, showmedians=False,
                       showextrema=False)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(pastel_colors[i*3])
    pc.set_alpha(0.3)

# Overlay box plot
bp2 = ax2.boxplot(weather_data, positions=range(1, len(weather_types)+1),
                 notch=True, patch_artist=True, widths=0.3,
                 boxprops=dict(facecolor='white', alpha=0.8),
                 medianprops=dict(color=PASTEL_EASTER['coral'], linewidth=2),
                 whiskerprops=dict(color='#666', linewidth=1),
                 capprops=dict(color='#666', linewidth=1))

# Add outlier analysis
for i, data in enumerate(weather_data, 1):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
    
    if len(outliers) > 0:
        ax2.scatter([i]*len(outliers), outliers, color=PASTEL_EASTER['coral'],
                   s=50, alpha=0.6, marker='*', edgecolors='#666', linewidth=1)
        
        # Annotate outlier count
        ax2.text(i, ax2.get_ylim()[1] - 0.5, f'{len(outliers)} outliers',
                ha='center', fontsize=9, color=PASTEL_EASTER['coral'],
                fontweight='bold')

ax2.set_xticks(range(1, len(weather_types)+1))
ax2.set_xticklabels(weather_types)
ax2.set_xlabel('Weather Condition', fontsize=12, color='#666', fontweight='medium')
ax2.set_ylabel('Happiness Score', fontsize=12, color='#666', fontweight='medium')
ax2.set_title('Box Plot with Violin Overlay: Weather Impact on Happiness\nDistribution Comparison with Outlier Detection',
             fontsize=13, color='#444', fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

plt.tight_layout()
plt.show()

print("Advanced box plots complete. Techniques demonstrated:")
print("- Notched boxes for confidence intervals")
print("- Custom outlier detection and annotation")
print("- Violin overlay for distribution shape")
print("- Statistical annotations (IQR, sample size)")
print("- Mean markers with legend")

# Cell 12: Polar Plot and Radar Chart
"""
Polar plots excel at showing cyclical patterns and multi-dimensional comparisons.
This creates both a polar area chart and a radar chart with custom styling.
The pastel colors create gentle radial gradients.
"""

fig = plt.figure(figsize=(16, 8), facecolor='#FAFAFA')

# Left: Polar area chart for time-based patterns
ax1 = fig.add_subplot(121, projection='polar')

# Create hourly aggregation (simulated 24-hour pattern)
hours = np.arange(24)
theta = hours * (2 * np.pi / 24)

# Simulate visitor patterns throughout the day
morning_peak = 9
evening_peak = 18
visitor_pattern = 30 + 20 * np.exp(-((hours - morning_peak)**2) / 10) + \
                  15 * np.exp(-((hours - evening_peak)**2) / 8) + \
                  np.random.normal(0, 3, 24)

# Create bars with gradient colors
colors_hour = [pastel_cmap(i/24) for i in range(24)]
bars = ax1.bar(theta, visitor_pattern, width=2*np.pi/24, bottom=20,
              color=colors_hour, alpha=0.8, edgecolor=PASTEL_EASTER['periwinkle'],
              linewidth=1.5)

# Add radial grid lines
ax1.set_ylim(0, 80)
ax1.set_yticks([20, 40, 60, 80])
ax1.set_yticklabels(['20', '40', '60', '80'], fontsize=9, color='#666')

# Set hour labels
ax1.set_xticks(theta)
ax1.set_xticklabels([f'{h:02d}:00' if h % 3 == 0 else '' for h in hours])

# Add annotations for peaks
for peak_hour, peak_name in [(morning_peak, 'Morning\nPeak'), (evening_peak, 'Evening\nPeak')]:
    peak_theta = peak_hour * (2 * np.pi / 24)
    peak_value = visitor_pattern[peak_hour]
    ax1.annotate(peak_name, xy=(peak_theta, peak_value + 20),
                xytext=(peak_theta, peak_value + 30),
                ha='center', fontsize=10, color='#444', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor=PASTEL_EASTER['butter'], alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1))

ax1.set_title('24-Hour Visitor Pattern: Polar Area Chart\nCircadian Rhythm of Garden Visits',
             fontsize=13, color='#444', fontweight='bold', pad=20, y=1.1)
ax1.grid(True, alpha=0.3, color=PASTEL_EASTER['lavender'])

# Right: Radar chart for multi-dimensional comparison
ax2 = fig.add_subplot(122, projection='polar')

# Categories for radar chart
categories = ['Temperature\nAdaptability', 'Rainfall\nTolerance', 'Growth\nRate', 
             'Visitor\nAttraction', 'Happiness\nIndex', 'Maintenance\nNeed']
n_cats = len(categories)

# Create data for three different plant types (simulated)
plant_types = ['Spring Flowers', 'Summer Herbs', 'Shade Plants']
plant_data = {
    'Spring Flowers': [8, 6, 9, 10, 9, 4],
    'Summer Herbs': [9, 4, 7, 8, 8, 6],
    'Shade Plants': [5, 8, 5, 6, 7, 8]
}

# Set up angles for radar chart
angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Plot each plant type
colors_radar = [PASTEL_EASTER['mint'], PASTEL_EASTER['coral'], PASTEL_EASTER['sky']]
for i, (plant, values) in enumerate(plant_data.items()):
    values += values[:1]  # Complete the circle
    
    # Plot area
    ax2.plot(angles, values, 'o-', linewidth=2, color=colors_radar[i], label=plant)
    ax2.fill(angles, values, alpha=0.3, color=colors_radar[i])
    
    # Add value labels
    for angle, value in zip(angles[:-1], values[:-1]):
        ax2.text(angle, value + 0.5, str(value), ha='center', va='center',
                fontsize=9, color=colors_radar[i], fontweight='bold')

# Customize radar chart
ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontsize=10, color='#444')
ax2.set_ylim(0, 10)
ax2.set_yticks([2, 4, 6, 8, 10])
ax2.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9, color='#666')
ax2.set_title('Plant Type Comparison: Radar Chart\nMulti-dimensional Performance Analysis',
             fontsize=13, color='#444', fontweight='bold', pad=30, y=1.15)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fancybox=True)
ax2.grid(True, alpha=0.3, color=PASTEL_EASTER['lavender'])

plt.tight_layout()
plt.show()

print("Polar and radar charts complete. Techniques demonstrated:")
print("- Polar area chart for temporal patterns")
print("- Peak detection and annotation")
print("- Radar chart for multi-dimensional comparison")
print("- Custom angle positioning and labels")
print("- Gradient colors in polar coordinates")

# Cell 13: Histogram with KDE and Statistical Overlays
"""
Histograms reveal distribution shapes. This advanced version includes
kernel density estimation, multiple distribution overlays, and statistical
annotations. The pastel colors create layered transparency.
"""

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='#FAFAFA')

# Top Left: Histogram with KDE overlay
data = garden_data['happiness_score'].values

n, bins, patches = ax1.hist(data, bins=20, density=True, alpha=0.7,
                           color=PASTEL_EASTER['mint'], edgecolor=PASTEL_EASTER['periwinkle'],
                           linewidth=1.5, label='Observed')

# Color bars by height (gradient effect)
max_height = max([p.get_height() for p in patches])
for patch in patches:
    height = patch.get_height()
    color_intensity = height / max_height
    patch.set_facecolor(pastel_cmap(color_intensity))

# Add KDE overlay
kde = gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 200)
kde_values = kde(x_range)
ax1.plot(x_range, kde_values, color=PASTEL_EASTER['coral'], linewidth=3,
        label='KDE', alpha=0.8)

# Add normal distribution overlay
mu, sigma = data.mean(), data.std()
normal_dist = stats.norm(mu, sigma)
ax1.plot(x_range, normal_dist.pdf(x_range), '--', color=PASTEL_EASTER['periwinkle'],
        linewidth=2, label='Normal Fit', alpha=0.8)

# Statistical annotations
ax1.axvline(mu, color=PASTEL_EASTER['coral'], linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(np.median(data), color=PASTEL_EASTER['sky'], linestyle='--', linewidth=2, alpha=0.7)

# Add text box with statistics
stats_text = f'Mean: {mu:.2f}\nMedian: {np.median(data):.2f}\nStd: {sigma:.2f}\nSkew: {stats.skew(data):.2f}\nKurtosis: {stats.kurtosis(data):.2f}'
ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
        facecolor=PASTEL_EASTER['lavender'], alpha=0.3))

ax1.set_xlabel('Happiness Score', fontsize=11, color='#666')
ax1.set_ylabel('Density', fontsize=11, color='#666')
ax1.set_title('Distribution Analysis: Histogram with KDE and Normal Fit\nStatistical Shape Assessment',
             fontsize=12, color='#444', fontweight='bold', pad=15)
ax1.legend(loc='upper right', frameon=True, fancybox=True)
ax1.grid(axis='y', alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

# Top Right: Stacked histogram for categories
weather_happy = {weather: garden_data[garden_data['weather'] == weather]['happiness_score'].values 
                for weather in weather_types}

ax2.hist([weather_happy[w] for w in weather_types], bins=15, stacked=True,
        color=[PASTEL_EASTER['mint'], PASTEL_EASTER['sky'], PASTEL_EASTER['coral']],
        alpha=0.8, edgecolor=PASTEL_EASTER['periwinkle'], linewidth=1,
        label=weather_types)

# Add cumulative line
cumulative_data = np.sort(garden_data['happiness_score'].values)
cumulative_prob = np.arange(1, len(cumulative_data) + 1) / len(cumulative_data)

ax2_twin = ax2.twinx()
ax2_twin.plot(cumulative_data, cumulative_prob * 100, color=PASTEL_EASTER['periwinkle'],
             linewidth=2, label='Cumulative %', alpha=0.8)
ax2_twin.set_ylabel('Cumulative Percentage (%)', fontsize=11, color='#666')

ax2.set_xlabel('Happiness Score', fontsize=11, color='#666')
ax2.set_ylabel('Frequency', fontsize=11, color='#666')
ax2.set_title('Stacked Histogram by Weather Condition\nWith Cumulative Distribution Function',
             fontsize=12, color='#444', fontweight='bold', pad=15)
ax2.legend(loc='upper left', frameon=True, fancybox=True)
ax2.grid(axis='y', alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

# Bottom Left: Step histogram with confidence bands
ax3.hist(data, bins=20, histtype='step', linewidth=2, color=PASTEL_EASTER['coral'],
        alpha=0.8, label='Step Histogram')

# Bootstrap confidence intervals
n_bootstrap = 100
bootstrap_hists = []
for _ in range(n_bootstrap):
    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
    hist_vals, _ = np.histogram(bootstrap_sample, bins=20)
    bootstrap_hists.append(hist_vals)

bootstrap_hists = np.array(bootstrap_hists)
lower_bound = np.percentile(bootstrap_hists, 2.5, axis=0)
upper_bound = np.percentile(bootstrap_hists, 97.5, axis=0)

# Plot confidence bands
bin_centers = (bins[:-1] + bins[1:]) / 2
ax3.fill_between(bin_centers, lower_bound, upper_bound, alpha=0.3,
                color=PASTEL_EASTER['sky'], label='95% CI')

ax3.set_xlabel('Happiness Score', fontsize=11, color='#666')
ax3.set_ylabel('Frequency', fontsize=11, color='#666')
ax3.set_title('Step Histogram with Bootstrap Confidence Intervals\nUncertainty Quantification',
             fontsize=12, color='#444', fontweight='bold', pad=15)
ax3.legend(loc='upper right', frameon=True, fancybox=True)
ax3.grid(axis='y', alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

# Bottom Right: 2D histogram (hexbin)
x_hex = garden_data['temperature'].values
y_hex = garden_data['happiness_score'].values

hexbin = ax4.hexbin(x_hex, y_hex, gridsize=15, cmap=pastel_cmap, alpha=0.8,
                    edgecolors=PASTEL_EASTER['periwinkle'], linewidths=0.5)

# Add marginal rug plots
ax4.plot(x_hex, [ax4.get_ylim()[0]] * len(x_hex), '|', color=PASTEL_EASTER['coral'],
        alpha=0.3, markersize=8)
ax4.plot([ax4.get_xlim()[0]] * len(y_hex), y_hex, '_', color=PASTEL_EASTER['mint'],
        alpha=0.3, markersize=8)

ax4.set_xlabel('Temperature (°C)', fontsize=11, color='#666')
ax4.set_ylabel('Happiness Score', fontsize=11, color='#666')
ax4.set_title('2D Hexbin Histogram: Temperature vs Happiness\nWith Marginal Rug Plots',
             fontsize=12, color='#444', fontweight='bold', pad=15)

# Add colorbar
cbar = plt.colorbar(hexbin, ax=ax4)
cbar.set_label('Count', fontsize=10, color='#666')

ax4.grid(True, alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

plt.tight_layout()
plt.show()

print("Advanced histograms complete. Techniques demonstrated:")
print("- KDE and normal distribution overlays")
print("- Stacked histograms with categories")
print("- Cumulative distribution function")
print("- Bootstrap confidence intervals")
print("- 2D hexbin histograms with rug plots")

# Cell 14: Stream Plot and Area Chart
"""
Stream plots show flow and area charts show cumulative contributions.
This creates both a streamplot for vector fields and stacked area chart
for time series decomposition. The pastel gradient creates smooth flows.
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='#FAFAFA')

# Left: Streamplot showing garden "flow" patterns
x_stream = np.linspace(0, 10, 20)
y_stream = np.linspace(0, 10, 20)
X_stream, Y_stream = np.meshgrid(x_stream, y_stream)

# Create vector field (simulating air flow in garden)
U = -1 - X_stream + Y_stream**2
V = 1 + X_stream - 2*Y_stream

# Calculate speed for coloring
speed = np.sqrt(U**2 + V**2)

# Create streamplot
stream = ax1.streamplot(X_stream, Y_stream, U, V, color=speed, cmap=pastel_cmap,
                        linewidth=2, density=1.5, arrowstyle='->', arrowsize=1.5,
                        minlength=0.1, maxlength=4.0)

# Add start points
start_points = [(1, 1), (9, 1), (5, 9)]
for x_start, y_start in start_points:
    ax1.plot(x_start, y_start, 'o', color=PASTEL_EASTER['coral'], markersize=10,
            markeredgecolor='white', markeredgewidth=2)
    ax1.annotate('Source', xy=(x_start, y_start), xytext=(x_start+0.5, y_start+0.5),
                fontsize=9, color='#444', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Add sink point
ax1.plot(5, 5, 's', color=PASTEL_EASTER['periwinkle'], markersize=12,
        markeredgecolor='white', markeredgewidth=2)
ax1.annotate('Sink', xy=(5, 5), xytext=(6, 6),
            fontsize=9, color='#444', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_xlabel('Garden Width (m)', fontsize=11, color='#666')
ax1.set_ylabel('Garden Length (m)', fontsize=11, color='#666')
ax1.set_title('Air Flow Patterns in Garden: Streamplot Visualization\nVector Field Analysis',
             fontsize=12, color='#444', fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

# Add colorbar
cbar = plt.colorbar(stream.lines, ax=ax1)
cbar.set_label('Flow Speed', fontsize=10, color='#666')

# Right: Stacked area chart
dates = garden_data['date']
components = {
    'Base Growth': np.ones(len(dates)) * 20,
    'Temperature Effect': np.maximum(0, (garden_data['temperature'] - 15) * 2),
    'Rainfall Effect': np.maximum(0, (garden_data['rainfall'] - 10) * 0.5),
    'Seasonal Boost': np.where(garden_data['season'] == 'Mid Spring', 10, 0)
}

# Create stacked area
ax2.stackplot(dates, components.values(),
             labels=components.keys(),
             colors=[PASTEL_EASTER['mint'], PASTEL_EASTER['sky'], 
                    PASTEL_EASTER['butter'], PASTEL_EASTER['coral']],
             alpha=0.8)

# Add total line
total = sum(components.values())
ax2.plot(dates, total, color=PASTEL_EASTER['periwinkle'], linewidth=2.5,
        label='Total Effect', alpha=0.9)

# Add annotations for significant changes
change_points = [20, 45, 70]
for idx in change_points:
    ax2.axvline(dates.iloc[idx], color='#666', linestyle='--', alpha=0.5)
    
    # Calculate composition at this point
    composition = {k: v[idx] for k, v in components.items()}
    dominant = max(composition, key=composition.get)
    
    ax2.annotate(f'Dominant:\n{dominant}', xy=(dates.iloc[idx], total[idx]),
                xytext=(dates.iloc[idx] + pd.Timedelta(days=3), total[idx] + 5),
                fontsize=9, color='#444',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor=PASTEL_EASTER['lavender'], alpha=0.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1))

ax2.set_xlabel('Date', fontsize=11, color='#666')
ax2.set_ylabel('Growth Components', fontsize=11, color='#666')
ax2.set_title('Growth Factor Decomposition: Stacked Area Chart\nTemporal Contribution Analysis',
             fontsize=12, color='#444', fontweight='bold', pad=15)
ax2.legend(loc='upper left', frameon=True, fancybox=True)
ax2.grid(axis='y', alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])

# Format x-axis
ax2.set_xlim(dates.min(), dates.max())
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("Stream and area plots complete. Techniques demonstrated:")
print("- Vector field visualization with streamlines")
print("- Flow speed coloring")
print("- Source and sink annotations")
print("- Stacked area chart for decomposition")
print("- Change point detection and annotation")

# Cell 15: Waterfall Chart for Sequential Changes
"""
Waterfall charts excel at showing sequential changes and their cumulative effect.
This creates an advanced waterfall with color coding, connecting lines, and
annotations. The pastel colors distinguish positive and negative changes.
"""

fig, ax = plt.subplots(figsize=(14, 8), facecolor='#FAFAFA')

# Create data for waterfall chart
categories = ['Initial\nValue', 'Spring\nRains', 'Temperature\nRise', 'Fertilizer\nAdded',
             'Pest\nDamage', 'Recovery\nGrowth', 'Late\nFrost', 'Final\nRecovery', 'End\nValue']

values = [50, 15, 20, 25, -10, 12, -8, 15, 0]  # Last value will be calculated
values[-1] = sum(values[:-1])  # Calculate final value

# Calculate positions
cumulative = [values[0]]
for v in values[1:-1]:
    cumulative.append(cumulative[-1] + v)
cumulative.append(values[-1])

# Create bars
for i, (cat, val, cum) in enumerate(zip(categories, values, cumulative)):
    if i == 0:  # Starting value
        color = PASTEL_EASTER['periwinkle']
        bottom = 0
        height = val
    elif i == len(categories) - 1:  # Final value
        color = PASTEL_EASTER['periwinkle']
        bottom = 0
        height = cum
    else:  # Changes
        if val >= 0:
            color = PASTEL_EASTER['mint']
            bottom = cumulative[i-1]
            height = val
        else:
            color = PASTEL_EASTER['coral']
            bottom = cumulative[i]
            height = abs(val)
    
    bar = ax.bar(i, height, bottom=bottom, color=color, alpha=0.8,
                edgecolor=PASTEL_EASTER['periwinkle'], linewidth=1.5)
    
    # Add value labels
    label_y = bottom + height/2 if height > 0 else bottom - abs(height)/2
    ax.text(i, label_y, f'{val:+.0f}' if i != 0 and i != len(categories)-1 else f'{cum:.0f}',
           ha='center', va='center', fontsize=11, color='white' if abs(height) > 10 else '#444',
           fontweight='bold')
    
    # Add percentage change
    if i > 0 and i < len(categories) - 1:
        pct_change = (val / cumulative[i-1]) * 100
        ax.text(i, cumulative[i] + 3, f'{pct_change:+.1f}%', ha='center', fontsize=9,
               color='#666', fontstyle='italic')

# Add connecting lines
for i in range(len(cumulative) - 1):
    if i < len(cumulative) - 2:  # Don't connect to final bar
        ax.plot([i + 0.4, i + 1 - 0.4], [cumulative[i], cumulative[i]],
               'k--', alpha=0.5, linewidth=1)

# Add reference line at zero
ax.axhline(y=0, color='#666', linestyle='-', linewidth=1, alpha=0.5)

# Add cumulative line
ax.plot(range(len(categories)-1), cumulative[:-1], 'o-', color=PASTEL_EASTER['butter'],
       linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=2,
       alpha=0.9, label='Cumulative Total')

# Styling
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=10, color='#444')
ax.set_xlabel('Growth Factors', fontsize=12, color='#666', fontweight='medium')
ax.set_ylabel('Plant Growth Impact (cm)', fontsize=12, color='#666', fontweight='medium')
ax.set_title('Waterfall Chart: Sequential Growth Factor Analysis\nCumulative Impact Visualization',
            fontsize=13, color='#444', fontweight='bold', pad=15)

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=PASTEL_EASTER['mint'], alpha=0.8, label='Positive Impact'),
    Patch(facecolor=PASTEL_EASTER['coral'], alpha=0.8, label='Negative Impact'),
    Patch(facecolor=PASTEL_EASTER['periwinkle'], alpha=0.8, label='Total Values')
]
ax.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=True)

ax.grid(axis='y', alpha=0.3, linestyle='--', color=PASTEL_EASTER['lavender'])
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

print("Waterfall chart complete. Techniques demonstrated:")
print("- Sequential change visualization")
print("- Color coding for positive/negative impacts")
print("- Connecting lines between bars")
print("- Percentage change annotations")
print("- Cumulative total tracking")

# Cell 16: Final Summary Dashboard
"""
Bringing it all together in a comprehensive dashboard that showcases
multiple visualization types in a cohesive layout. This demonstrates
advanced subplot management and visual hierarchy using our pastel palette.
"""

fig = plt.figure(figsize=(20, 12), facecolor='#FAFAFA')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Spring Garden Analytics Dashboard: Complete Visualization Suite\nBy Cazzy Apporbo, 2025',
            fontsize=16, color='#444', fontweight='bold', y=0.98)

# 1. Time series mini plot
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(garden_data['date'], garden_data['plant_growth'], 
        color=PASTEL_EASTER['periwinkle'], linewidth=2)
ax1.fill_between(garden_data['date'], 0, garden_data['plant_growth'],
                 alpha=0.3, color=PASTEL_EASTER['mint'])
ax1.set_title('Growth Timeline', fontsize=11, color='#444', fontweight='bold')
ax1.set_ylabel('Growth (cm)', fontsize=9, color='#666')
ax1.tick_params(axis='x', rotation=45, labelsize=8)
ax1.grid(True, alpha=0.3, color=PASTEL_EASTER['lavender'])

# 2. Correlation heatmap mini
ax2 = fig.add_subplot(gs[0, 2])
mini_corr = garden_data[['temperature', 'rainfall', 'happiness_score']].corr()
im = ax2.imshow(mini_corr, cmap=pastel_cmap, aspect='auto', vmin=-1, vmax=1)
ax2.set_xticks(range(3))
ax2.set_yticks(range(3))
ax2.set_xticklabels(['Temp', 'Rain', 'Happy'], fontsize=8, rotation=45)
ax2.set_yticklabels(['Temp', 'Rain', 'Happy'], fontsize=8)
ax2.set_title('Correlations', fontsize=11, color='#444', fontweight='bold')

# 3. Distribution plot
ax3 = fig.add_subplot(gs[0, 3])
ax3.hist(garden_data['happiness_score'], bins=15, color=PASTEL_EASTER['coral'],
        alpha=0.7, edgecolor=PASTEL_EASTER['periwinkle'])
ax3.set_title('Happiness Distribution', fontsize=11, color='#444', fontweight='bold')
ax3.set_xlabel('Score', fontsize=9, color='#666')
ax3.set_ylabel('Frequency', fontsize=9, color='#666')

# 4. Scatter with regression
ax4 = fig.add_subplot(gs[1, :2])
sc = ax4.scatter(garden_data['temperature'], garden_data['visitors'],
                c=garden_data['happiness_score'], cmap=pastel_cmap,
                s=50, alpha=0.6, edgecolors=PASTEL_EASTER['periwinkle'])
z = np.polyfit(garden_data['temperature'], garden_data['visitors'], 1)
p = np.poly1d(z)
ax4.plot(garden_data['temperature'], p(garden_data['temperature']),
        '--', color=PASTEL_EASTER['coral'], alpha=0.8)
ax4.set_title('Temperature vs Visitors', fontsize=11, color='#444', fontweight='bold')
ax4.set_xlabel('Temperature (°C)', fontsize=9, color='#666')
ax4.set_ylabel('Visitors', fontsize=9, color='#666')
plt.colorbar(sc, ax=ax4, label='Happiness', fraction=0.046, pad=0.04)

# 5. Box plot comparison
ax5 = fig.add_subplot(gs[1, 2:])
season_happiness = [garden_data[garden_data['season'] == s]['happiness_score'].values
                   for s in seasons]
bp = ax5.boxplot(season_happiness, labels=seasons, patch_artist=True,
                notch=True, widths=0.6)
for patch, color in zip(bp['boxes'], [PASTEL_EASTER['mint'], PASTEL_EASTER['sky'], PASTEL_EASTER['coral']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_title('Seasonal Happiness Comparison', fontsize=11, color='#444', fontweight='bold')
ax5.set_ylabel('Happiness Score', fontsize=9, color='#666')
ax5.grid(axis='y', alpha=0.3, color=PASTEL_EASTER['lavender'])

# 6. Pie chart
ax6 = fig.add_subplot(gs[2, 0])
weather_dist = garden_data['weather'].value_counts()
colors_pie = [PASTEL_EASTER['mint'], PASTEL_EASTER['butter'], PASTEL_EASTER['coral']]
wedges, texts, autotexts = ax6.pie(weather_dist.values, labels=weather_dist.index,
                                   colors=colors_pie, autopct='%1.1f%%',
                                   startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(8)
    autotext.set_fontweight('bold')
ax6.set_title('Weather Distribution', fontsize=11, color='#444', fontweight='bold')

# 7. Bar chart with error
ax7 = fig.add_subplot(gs[2, 1:3])
mean_growth = garden_data.groupby('season')['plant_growth'].agg(['mean', 'std'])
x_pos = np.arange(len(seasons))
bars = ax7.bar(x_pos, mean_growth['mean'].values,
              yerr=mean_growth['std'].values,
              capsize=5, color=[PASTEL_EASTER['mint'], PASTEL_EASTER['sky'], PASTEL_EASTER['coral']],
              alpha=0.8, edgecolor=PASTEL_EASTER['periwinkle'], linewidth=1.5)
ax7.set_xticks(x_pos)
ax7.set_xticklabels(seasons, fontsize=9)
ax7.set_title('Mean Growth by Season', fontsize=11, color='#444', fontweight='bold')
ax7.set_ylabel('Growth (cm)', fontsize=9, color='#666')
ax7.grid(axis='y', alpha=0.3, color=PASTEL_EASTER['lavender'])

# 8. Summary statistics text
ax8 = fig.add_subplot(gs[2, 3])
ax8.axis('off')
summary_stats = f"""
SUMMARY STATISTICS

Total Days: {len(garden_data)}
Avg Temperature: {garden_data['temperature'].mean():.1f}°C
Total Rainfall: {garden_data['rainfall'].sum():.0f}mm
Max Growth: {garden_data['plant_growth'].max():.1f}cm
Avg Visitors: {garden_data['visitors'].mean():.0f}
Happiness: {garden_data['happiness_score'].mean():.2f}/10

Peak Season: {garden_data.groupby('season')['happiness_score'].mean().idxmax()}
Best Weather: {garden_data.groupby('weather')['happiness_score'].mean().idxmax()}
"""
ax8.text(0.1, 0.9, summary_stats, transform=ax8.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=PASTEL_EASTER['lavender'], alpha=0.3))

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("COMPREHENSIVE VISUALIZATION TUTORIAL COMPLETE")
print("="*60)
print("\nTechniques Mastered:")
print("- 15+ different plot types demonstrated")
print("- Consistent pastel easter ombre aesthetic throughout")
print("- Advanced statistical annotations and overlays")
print("- Multi-dimensional data encoding")
print("- Animation and interactivity")
print("- Complex subplot layouts with GridSpec")
print("- Custom color gradients and transparency")
print("- Publication-quality labeling and styling")
print("\nKey Takeaways:")
print("- Always consider your color palette for visual harmony")
print("- Layer information using transparency and size")
print("- Annotate intelligently to guide interpretation")
print("- Combine plot types for comprehensive analysis")
print("- Maintain consistency in styling across visualizations")
print("\nThank you for learning advanced visualization with Cazzy Apporbo!")
print("May your data always be beautiful and insightful!")