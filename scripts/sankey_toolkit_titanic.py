"""
Exploratory Data Visualization and Transformation Toolkit
Author: Cazzy
Purpose: This toolkit demonstrates advanced Python capabilities, combining data manipulation,
functional transformations, backpropagation-style reshaping, and multi-dimensional visualization.
Each visualization and transformation shows a different skill, from classical aggregation to
interactive 3D plotting, Sankey flows, and creative data translation. The Titanic dataset
is used to illustrate these techniques, but the code is generalizable to other datasets.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Load Titanic dataset
titanic = sns.load_dataset("titanic")

# Pastel color palette for all visualizations
pastel_colors = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#D0BAFF", "#FFBAE1"]

# -------------------------------
# Data reshaping and backpropagation-style example
# -------------------------------
def backprop_style_transformation(df):
    df_copy = df.copy()
    # Fill missing values
    df_copy['age'] = df_copy['age'].fillna(df_copy['age'].median())
    # Normalize numeric columns
    numeric_cols = df_copy.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        col_mean = df_copy[col].mean()
        col_std = df_copy[col].std()
        df_copy[col] = (df_copy[col] - col_mean) / col_std
    # Create feature interactions (pseudo-backprop-style transformations)
    df_copy['age_fare_interaction'] = df_copy['age'] * df_copy['fare'].replace(np.nan, 0)
    return df_copy

# -------------------------------
# Functional programming example
# -------------------------------
def functional_group_summary(df):
    # Group by class and survived, aggregate multiple metrics using lambdas
    grouped = df.groupby(['class','survived']).agg(
        mean_age=('age', lambda x: round(x.mean(),2)),
        total_fare=('fare', lambda x: round(x.sum(),2)),
        passengers=('survived', 'count')
    ).reset_index()
    return grouped

# -------------------------------
# Sankey diagram demonstration
# -------------------------------
def sankey_demo(df):
    labels = []
    for col in ['class','sex','survived']:
        labels.extend(df[col].unique().tolist())
    labels = list(dict.fromkeys(labels))
    
    def get_index(val):
        return labels.index(val)
    
    sources = []
    targets = []
    values = []
    
    for col_from, col_to in [('class','sex'),('sex','survived')]:
        grouped = df.groupby([col_from,col_to]).size().reset_index(name='count')
        for _, row in grouped.iterrows():
            sources.append(get_index(row[col_from]))
            targets.append(get_index(row[col_to]))
            values.append(row['count'])
    
    colors = [pastel_colors[i % len(pastel_colors)] for i in range(len(labels))]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=labels,
            color=colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='lightgray'
        )
    )])
    
    fig.update_layout(title_text='Titanic Passenger Flow: Class to Gender to Survived', font_size=12)
    fig.show()

# -------------------------------
# Sunburst chart demonstration
# -------------------------------
def sunburst_demo(df):
    df['age_group'] = pd.cut(df['age'], bins=[0,12,18,35,60,100], labels=['Child','Teen','Adult','MidAge','Senior'])
    fig = px.sunburst(df, path=['class','age_group','survived'], values=None, color='survived',
                      color_discrete_sequence=pastel_colors)
    fig.update_layout(title_text='Titanic Sunburst: Class to Age Group to Survived')
    fig.show()

# -------------------------------
# Treemap demonstration
# -------------------------------
def treemap_demo(df):
    fig = px.treemap(df, path=['class','sex','survived'], values=None, color='survived',
                     color_discrete_sequence=pastel_colors)
    fig.update_layout(title_text='Titanic Treemap: Class to Sex to Survived')
    fig.show()

# -------------------------------
# 3D Scatter demonstration
# -------------------------------
def scatter3d_demo(df):
    df_clean = df.copy()
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    df_clean['fare'] = df_clean['fare'].fillna(0)
    fig = px.scatter_3d(df_clean, x='age', y='fare', z='pclass', color='survived',
                        color_discrete_sequence=pastel_colors, size='fare', hover_data=['sex'])
    fig.update_layout(title_text='3D Scatter: Age, Fare, Pclass with Survival')
    fig.show()

# -------------------------------
# Flow/stacked bar demonstration
# -------------------------------
def flow_demo(df):
    grouped = df.groupby(['class','survived']).size().reset_index(name='count')
    fig = px.bar(grouped, x='class', y='count', color='survived', barmode='stack',
                 color_discrete_sequence=pastel_colors)
    fig.update_layout(title_text='Titanic Stacked Flow: Class to Survived')
    fig.show()

# -------------------------------
# Main execution
# -------------------------------
def main():
    print('Running backprop-style transformation...')
    df_transformed = backprop_style_transformation(titanic)
    
    print('Running functional group summary...')
    summary = functional_group_summary(df_transformed)
    print(summary)
    
    print('Running Sankey demo...')
    sankey_demo(df_transformed)
    
    print('Running Sunburst demo...')
    sunburst_demo(df_transformed)
    
    print('Running Treemap demo...')
    treemap_demo(df_transformed)
    
    print('Running 3D scatter demo...')
    scatter3d_demo(df_transformed)
    
    print('Running Flow demo...')
    flow_demo(df_transformed)
    
    print('All visualizations and transformations executed successfully.')

if __name__ == '__main__':
    main()