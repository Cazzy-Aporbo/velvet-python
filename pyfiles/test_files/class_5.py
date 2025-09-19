import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the characters data file
chars = pd.read_csv('https://nces.ed.gov/ipeds/datacenter/data/HD2021.zip', compression='zip', encoding="ISO-8859-1")
# Retain selected columns from chars
chars = chars[['INSTNM', 'STABBR', 'CONTROL', 'UNITID']]

# Read the enrollment data file
enr = pd.read_csv('https://nces.ed.gov/ipeds/datacenter/data/EFFY2021.zip', compression='zip', encoding="ISO-8859-1")
# Retain selected columns and filter for EFFYALEV = 1
enr = enr[enr['EFFYALEV'] == 1][['EFYTOTLT', 'UNITID']]

# Drop CONTROL values equal to -3
chars = chars[chars['CONTROL'] != -3]

# Perform an inner join on UNITID to merge the two DataFrames
merged_data = pd.merge(chars, enr, on='UNITID', how='inner')

# Listwise drop records with any null values
merged_data.dropna(inplace=True)

# Ask user for color preference
color_preference = input("Enter color preference (pastel/standard): ")

if color_preference.lower() == 'pastel':
    hist_color = 'lightblue'
    cdf_color = 'mediumorchid'
    hist_subplot_colors = ['lightblue', 'lightgreen', 'lightcoral']
    line_color = 'mediumorchid'
else:
    hist_color = 'yellowgreen'
    cdf_color = 'darkcyan'
    hist_subplot_colors = ['springgreen', 'midnightblue', 'mediumvioletred']
    line_color = 'mediumorchid'

# Create subplots with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Plot histogram of total enrollment
sns.histplot(data=merged_data, x='EFYTOTLT', ax=ax1, bins=30, kde=False, color=hist_color)
ax1.set_title('Enrollment Distribution')
ax1.set_ylabel('Count')

# Overlay the cumulative distribution function (CDF)
sns.kdeplot(data=merged_data, x='EFYTOTLT', ax=ax2, cumulative=True, color=cdf_color)
ax2.set_xlabel('Enrollment')
ax2.set_ylabel('Cumulative Probability')
ax2.set_title('Enrollment CDF')

# Adjust layout and spacing
plt.tight_layout()

# Show the plot
plt.show()

# Create subplots for enrollment distributions by control
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Filter data by control type
public_data = merged_data[merged_data['CONTROL'] == 1]
private_data = merged_data[merged_data['CONTROL'] == 2]
for_profit_data = merged_data[merged_data['CONTROL'] == 3]

# Plot histograms for each control type
sns.histplot(data=public_data, x='EFYTOTLT', bins=30, kde=False, color=hist_subplot_colors[0], ax=axes[0])
axes[0].set_title('Enrollment Distribution - Public Institutions')
axes[0].set_ylabel('Count')

sns.histplot(data=private_data, x='EFYTOTLT', bins=30, kde=False, color=hist_subplot_colors[1], ax=axes[1])
axes[1].set_title('Enrollment Distribution - Private Institutions')
axes[1].set_ylabel('Count')

sns.histplot(data=for_profit_data, x='EFYTOTLT', bins=30, kde=False, color=hist_subplot_colors[2], ax=axes[2])
axes[2].set_title('Enrollment Distribution - For-Profit Institutions')
axes[2].set_xlabel('Enrollment')
axes[2].set_ylabel('Count')

# Adjust layout and spacing
plt.tight_layout()

# Show the plot
plt.show()

# Filter data for Colorado institutions
colorado_data = merged_data[merged_data['STABBR'] == 'CO']

# Create a separate plot for Colorado institutions
plt.figure(figsize=(10, 6))
sns.histplot(data=colorado_data, x='EFYTOTLT', bins=30, kde=False, color=hist_color)
plt.title('Enrollment Distribution - Colorado Institutions')
plt.xlabel('Enrollment')
plt.ylabel('Count')

# Highlight DU's enrollment
du_enrollment = colorado_data[colorado_data['INSTNM'] == 'University of Denver']['EFYTOTLT'].values[0]
plt.axvline(du_enrollment, color=line_color, linestyle='--', label="DU's Enrollment")
plt.legend()

# Show the plot
plt.show()
