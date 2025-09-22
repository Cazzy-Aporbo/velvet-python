"""
Exploratory Sankey Demo
Author: Cazzy
Purpose: This demo visualizes the flow of passengers in the Titanic dataset using a 
Sankey diagram. The goal is to show the distribution from class to gender to survival 
outcome in a clear, visually appealing way. Colors are carefully chosen in pastel tones 
to make the diagram elegant, human-readable, and portfolio-ready.
"""

import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

# Load Titanic dataset from seaborn
titanic = sns.load_dataset("titanic")

# Preprocess for Sankey diagram
# Map categorical values to integers
label_list = []
for col in ["class", "sex", "survived"]:
    label_list.extend(titanic[col].unique().tolist())
labels = list(dict.fromkeys(label_list))  # remove duplicates

# Helper function to get indices
def get_index(val):
    return labels.index(val)

# Build source-target-value lists
sources = []
targets = []
values = []

for col_from, col_to in [("class", "sex"), ("sex", "survived")]:
    grouped = titanic.groupby([col_from, col_to]).size().reset_index(name="count")
    for _, row in grouped.iterrows():
        sources.append(get_index(row[col_from]))
        targets.append(get_index(row[col_to]))
        values.append(row["count"])

# Pastel color palette
pastel_colors = [
    "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#D0BAFF", "#FFBAE1"
]
colors = [pastel_colors[i % len(pastel_colors)] for i in range(len(labels))]

# Create Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color=colors
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color="lightgray"
    )
)])

fig.update_layout(
    title_text="Titanic Passenger Flow: Class → Gender → Survival",
    font_size=12
)

fig.show()