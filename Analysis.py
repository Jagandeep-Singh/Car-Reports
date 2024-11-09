#!/usr/bin/env python
# coding: utf-8

# Task 1: How does the distribution of car prices vary by brand and body style?
# 
# _Hints: Stacked column chart to show the distribution of car prices by brand and body style._

# Option 1: _matplotlib and seaborn_

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Read the CSV file
df = pd.read_csv('car_data.csv')

# 2. Group the data by brand and body style, calculating mean prices
grouped_data = df.groupby(['Make', 'Vehicle Style'])['MSRP'].mean().unstack()

# 3. Create the stacked column chart
plt.figure(figsize=(12, 6))
grouped_data.plot(kind='bar', stacked=True)

# 4. Customize the chart
plt.title('Average Car Prices by Brand and Body Style', fontsize=15)
plt.xlabel('Brand', fontsize=12)
plt.ylabel('Average Price', fontsize=12)

# Remove legend
plt.legend().remove()

# Reduce x-axis label size and rotate
plt.xticks(rotation=90, ha='center', fontsize=6)
plt.tight_layout()

# 5. Show the chart
plt.show()


# Option 2: _plotly_

# In[7]:


import pandas as pd
import plotly.graph_objs as go

# 1. Read the CSV file
try:
    df = pd.read_csv('car_data.csv')
except FileNotFoundError:
    print("The file 'car_data.csv' was not found.")
    exit()

# Check if necessary columns exist
required_columns = ['Make', 'Vehicle Style', 'MSRP']
if not all(col in df.columns for col in required_columns):
    print(f"Missing columns in the data: {set(required_columns) - set(df.columns)}")
    exit()

# Ensure 'MSRP' is numeric
df['MSRP'] = pd.to_numeric(df['MSRP'], errors='coerce')

# 2. Dropdown filter with multiple selection
def create_multi_select_chart(df):
    # Prepare data
    grouped_data = df.groupby(['Make', 'Vehicle Style'])['MSRP'].mean().reset_index()
    
    # Create figure with dropdown
    fig = go.Figure()

    # Add traces for each Vehicle Style
    vehicle_styles = grouped_data['Vehicle Style'].unique()
    for style in vehicle_styles:
        style_data = grouped_data[grouped_data['Vehicle Style'] == style]
        fig.add_trace(
            go.Bar(
                x=style_data['Make'], 
                y=style_data['MSRP'], 
                name=style,
                visible=(style == vehicle_styles[0])  # First style visible by default
            )
        )

    # Create visibility list for all styles
    visibility_all = [True] * len(vehicle_styles)

    # Update layout with dropdown
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                # Option to show all vehicle styles
                {
                    'method': 'update',
                    'label': 'All Vehicle Styles',
                    'args': [
                        {
                            'visible': visibility_all
                        },
                        {
                            'title': 'Average Car MSRP - All Vehicle Styles',
                            'barmode': 'stack'  # Set barmode to stack
                        }
                    ]
                }
            ] + [
                {
                    'method': 'update',
                    'label': style,
                    'args': [
                        {'visible': [style == trace.name for trace in fig.data]},
                        {'title': f'Average Car MSRP - {style} Vehicle Style', 'barmode': 'group'}  # Set barmode to group for individual styles
                    ]
                } 
                for style in vehicle_styles
            ],
            'direction': 'down',
            'showactive': True,
            'font': dict(size=10),  # Reduce font size of dropdown menu
        }],
        title='Car MSRP by Make and Vehicle Style',
        xaxis_title='Make',
        yaxis_title='Average MSRP',
        barmode='group',  # Default to group
        height=500,  # Height of the chart
        width=900,   # Width of the chart
        xaxis=dict(
            tickfont=dict(size=6),  # Reduce font size on x-axis
            tickangle=90,  # Rotate x-axis labels for better visibility
            automargin=True  # Automatically adjust margins
        ),
        margin=dict(l=50, r=50, t=50, b=100),  # Adjust margins to fit labels
        legend=dict(
            font=dict(size=8),  # Reduce font size of the legend
        )
    )

    return fig

# Generate the chart
multi_select_chart = create_multi_select_chart(df)

# Display the chart
multi_select_chart.show()


# Task 2: Which car brands have the highest and lowest average MSRPs, and how does this vary by body style?
# 
# _Hints: Clustered column chart to compare the average MSRPs across different car brands and body styles_

# Top 5 Brands

# In[8]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
try:
    df = pd.read_csv('car_data.csv')
except FileNotFoundError:
    print("Error: The file 'car_data.csv' was not found.")
    exit()

# Check for missing values in critical columns
if df[['Make', 'Vehicle Style', 'MSRP']].isnull().any().any():
    print("Warning: Missing values detected in the critical columns.")
    # You might want to handle missing values here

# Group by 'Make' and 'Vehicle Style' and calculate the average MSRP
average_msrp = df.groupby(['Make', 'Vehicle Style'])['MSRP'].mean().reset_index()

# Get the top 5 Makes based on average MSRP
top_5_makes = average_msrp.groupby('Make')['MSRP'].mean().nlargest(5).index
top_5_average_msrp = average_msrp[average_msrp['Make'].isin(top_5_makes)]

# Sort the values in ascending order for better visualization
top_5_average_msrp = top_5_average_msrp.sort_values(by='MSRP')

# Create a clustered column chart
plt.figure(figsize=(12, 6))
sns.barplot(data=top_5_average_msrp, x='Make', y='MSRP', hue='Vehicle Style', palette='viridis')

# Add title and labels
plt.title('Top 5 Car Brands by Average MSRP', fontsize=16)
plt.xlabel('Car Brand', fontsize=12)
plt.ylabel('Average MSRP', fontsize=12)
plt.xticks(rotation=45)  # Rotate x labels for better readability
plt.legend(title='Body Style')

# Show the plot
plt.tight_layout()
plt.show()


# Lowest 5 Brands

# In[18]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
try:
    df = pd.read_csv('car_data.csv')
except FileNotFoundError:
    print("Error: The file 'car_data.csv' was not found.")
    exit()

# Check for missing values in critical columns
if df[['Make', 'Vehicle Style', 'MSRP']].isnull().any().any():
    print("Warning: Missing values detected in the critical columns.")
    # You might want to handle missing values here

# Group by 'Make' and 'Vehicle Style' and calculate the average MSRP
average_msrp = df.groupby(['Make', 'Vehicle Style'])['MSRP'].mean().reset_index()

# Get the lowest 5 Makes based on average MSRP
lowest_5_makes = average_msrp.groupby('Make')['MSRP'].mean().nsmallest(5).index
lowest_5_average_msrp = average_msrp[average_msrp['Make'].isin(lowest_5_makes)]

# Sort the values in ascending order for better visualization
lowest_5_average_msrp = lowest_5_average_msrp.sort_values(by='MSRP')

# Create a clustered column chart
plt.figure(figsize=(12, 6))
sns.barplot(data=lowest_5_average_msrp, x='Make', y='MSRP', hue='Vehicle Style', palette='viridis')

# Add title and labels
plt.title('Lowest 5 Car Brands by Average MSRP', fontsize=16)
plt.xlabel('Car Brand', fontsize=12)
plt.ylabel('Average MSRP', fontsize=12)
plt.xticks(rotation=45)  # Rotate x labels for better readability
plt.legend(title='Body Style', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()


# Task 3: How do the different feature such as transmission type affect the MSRP, and how does this vary by body style?
# 
# _Hints: Scatter plot chart to visualize the relationship between MSRP and transmission type, with different symbols for each body style_

# In[ ]:


# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the CSV File
# Load the car_data CSV file
df = pd.read_csv('car_data.csv')

# Step 3: Data Preparation
# Ensure that 'MSRP' and 'Transmission Type' are in the correct format
df['MSRP'] = pd.to_numeric(df['MSRP'], errors='coerce')  # Convert MSRP to numeric
df['Transmission Type'] = df['Transmission Type'].astype(str)  # Ensure Transmission Type is a string

# Step 4: Create the Scatter Plot
plt.figure(figsize=(12, 8))

# Use Seaborn's scatterplot to plot MSRP vs. Transmission Type with different markers for each body style
sns.scatterplot(data=df, x='Transmission Type', y='MSRP', hue='Vehicle Style', style='Vehicle Style', s=100)

# Add titles and labels
plt.title('Relationship Between MSRP and Transmission Type by Vehicle Style')
plt.xlabel('Transmission Type')
plt.ylabel('MSRP ($)')
plt.xticks(rotation=45)
plt.legend(title='Vehicle Style', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()


# Task 4: How does the fuel efficiency of cars vary across different body styles and model years? 
# 
# _Hints: Line chart to show the trend of fuel efficiency (MPG) over time for each body style._

# Option 1: _matplotlib and seaborn_

# In[3]:


# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the CSV File
# Load the car_data CSV file
df = pd.read_csv('car_data.csv')

# Step 3: Calculate Average MPG
# Assuming the columns for highway MPG and city MPG are named 'highway MPG' and 'city mpg'
df['Average_MPG'] = (df['highway MPG'] + df['city mpg']) / 2

# Step 4: Sort DataFrame by Year
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')  # Ensure Year is numeric
df = df.sort_values(by='Year')  # Sort by Year in ascending order

# Step 5: Filter Vehicle Styles with More Than One Data Point
# Count the number of years for each vehicle style
vehicle_style_counts = df['Vehicle Style'].value_counts()
valid_vehicle_styles = vehicle_style_counts[vehicle_style_counts > 1].index.tolist()

# Filter the DataFrame to include only valid vehicle styles
filtered_df = df[df['Vehicle Style'].isin(valid_vehicle_styles)]

# Step 6: Create the Facet Grid with Seaborn
plt.figure(figsize=(16, 12))
g = sns.FacetGrid(filtered_df, col='Vehicle Style', col_wrap=4, height=4, sharey=False)
g.map(plt.plot, 'Year', 'Average_MPG')
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Year", "Average Fuel Efficiency (MPG)")
g.add_legend()

# Add a main title
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Trend of Average Fuel Efficiency (MPG) Over Time by Vehicle Style')

# Show the plot
plt.show()


# Option 2: _plotly_

# In[27]:


# Step 1: Import Required Libraries
import pandas as pd
import plotly.express as px

# Step 2: Load the CSV File
# Load the car_data CSV file
df = pd.read_csv('car_data.csv')

# Step 3: Calculate Average MPG
# Assuming the columns for highway MPG and city MPG are named 'highway MPG' and 'city mpg'
df['Average_MPG'] = (df['highway MPG'] + df['city mpg']) / 2

# Step 4: Sort DataFrame by Year
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')  # Ensure Year is numeric
df = df.sort_values(by='Year')  # Sort by Year in ascending order

# Step 5: Filter Vehicle Styles with More Than One Data Point
# Count the number of years for each vehicle style
vehicle_style_counts = df['Vehicle Style'].value_counts()
valid_vehicle_styles = vehicle_style_counts[vehicle_style_counts > 1].index.tolist()

# Filter the DataFrame to include only valid vehicle styles
filtered_df = df[df['Vehicle Style'].isin(valid_vehicle_styles)]

# Step 6: Create the Line Chart with Plotly
# Create a line plot with dropdown for filtering by year
fig = px.line(filtered_df, 
              x='Vehicle Style', 
              y='Average_MPG', 
              color='Vehicle Style', 
              line_group='Vehicle Style', 
              title='Trend of Average Fuel Efficiency (MPG) Over Time by Vehicle Style',
              animation_frame='Year', 
              range_y=[filtered_df['Average_MPG'].min(), filtered_df['Average_MPG'].max()])  # Autoscale y-axis

# Update layout for better readability
fig.update_layout(
    xaxis_title='Vehicle Style',
    yaxis_title='Average Fuel Efficiency (MPG)',
    xaxis_tickangle=-45,
    legend_title_text='Vehicle Style',
    updatemenus=[{
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'mode': 'immediate'}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]
)

# Show the plot
fig.show()


# Task 5: How does the car's horsepower, MPG, and price vary across different Brands?
# 
# _Hints: Bubble chart to visualize the relationship between horsepower, MPG, and price across different car brands. Assign different colors to each brand and label the bubbles with the brand name._

# Option 1: _Matplotib_

# In[66]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np

# Load the dataset
df = pd.read_csv('car_data.csv')

# Drop rows with NaNs in key columns to prevent calculation errors
df.dropna(subset=['highway MPG', 'city mpg', 'MSRP', 'Engine HP'], inplace=True)

# Calculate average MPG for each vehicle style
df['Average MPG'] = (df['highway MPG'] + df['city mpg']) / 2

# Calculate average values (MPG, MSRP, Engine HP) by brand
average_stats_by_brand = df.groupby('Make').agg(
    Average_MPG_by_Brand=('Average MPG', 'mean'),
    Average_MSRP=('MSRP', 'mean'),
    Average_Engine_HP=('Engine HP', 'mean')
).reset_index()

# Define a color palette with different shades
palette = sns.color_palette("husl", len(average_stats_by_brand))  # Using husl for distinct shades
color_map = dict(zip(average_stats_by_brand['Make'], palette))

# Create the bubble chart, using MSRP directly for bubble size
plt.figure(figsize=(14, 8))
sns.scatterplot(
    data=average_stats_by_brand,
    x='Average_Engine_HP',
    y='Average_MPG_by_Brand',
    size='Average_MSRP',
    sizes=(20, 1000),  # Adjust bubble size range
    hue='Make',
    palette=color_map,
    alpha=0.85,
    edgecolor='w',
    linewidth=0.5,
    legend=False  # Disable default legend
)

# Customize plot layout
plt.xlabel('Average Horsepower (Engine HP)')
plt.ylabel('Average MPG by Brand')
plt.title('Bubble Chart of Average Engine HP, Average MPG by Brand, and Average MSRP by Car Brand')

# Set axis limits dynamically
plt.xlim(0, average_stats_by_brand['Average_Engine_HP'].max() * 1.1)
plt.ylim(5, average_stats_by_brand['Average_MPG_by_Brand'].max() + 5)

# Reduce width gap between x-axis intervals
# Here, you can specify the ticks you want to display
x_ticks = np.arange(0, average_stats_by_brand['Average_Engine_HP'].max() * 1.1, 100)  # Adjust the step size (e.g., 20)
plt.xticks(ticks=x_ticks)

# Custom legend with brand colors
legend_elements = [Line2D([0], [0], marker='o', color='w', label=make,
                          markerfacecolor=color_map[make], markersize=10)
                   for make in average_stats_by_brand['Make']]
plt.legend(
    handles=legend_elements,
    title="Car Brand",
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=5,
    frameon=False
)

# Display the plot with gridlines
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout for legend space
plt.show()


# Option 2: _plotly_

# In[21]:


import pandas as pd
import plotly.express as px

# Step 1: Load the dataset
df = pd.read_csv('car_data.csv')

# Step 2: Calculate average MPG for each vehicle style
df['Average MPG'] = (df['highway MPG'] + df['city mpg']) / 2

# Step 3: Calculate average MPG for each brand
average_mpg_by_brand = df.groupby('Make')['Average MPG'].mean().reset_index()
average_mpg_by_brand.columns = ['Make', 'Average MPG by Brand']

# Step 4: Calculate average MSRP and average Engine HP by brand
average_msrp = df.groupby('Make')['MSRP'].mean().reset_index()
average_msrp.columns = ['Make', 'Average MSRP']

average_hp = df.groupby('Make')['Engine HP'].mean().reset_index()
average_hp.columns = ['Make', 'Average Engine HP']

# Step 5: Merge average values back into the original DataFrame
df = df.merge(average_msrp, on='Make', how='left')
df = df.merge(average_hp, on='Make', how='left')
df = df.merge(average_mpg_by_brand, on='Make', how='left')

# Step 6: Create the bubble chart
fig = px.scatter(df, 
                 x='Average Engine HP',  # Use average Engine HP for the x-axis
                 y='Average MPG by Brand',  # Use average MPG by brand for the y-axis
                 size='Average MSRP',  # Size based on average MSRP
                 color='Make', 
                 hover_name='Vehicle Style', 
                 title='Bubble Chart of Average Engine HP, Average MPG by Brand, and Average MSRP by Car Brand',
                 size_max=60)

# Step 7: Customize the layout
fig.update_layout(
    xaxis_title='Average Horsepower (Engine HP)',
    yaxis_title='Average MPG by Brand',
    legend_title='Car Brand',
    paper_bgcolor='rgba(243, 243, 243, 0.8)',
    plot_bgcolor='rgba(255, 255, 255, 0.8)'
)

# Step 8: Add gridlines
fig.update_xaxes(showgrid=True, gridcolor='LightGray')  # Add gridlines to x-axis
fig.update_yaxes(showgrid=True, gridcolor='LightGray')  # Add gridlines to y-axis

# Step 9: Show the plot
fig.show()

