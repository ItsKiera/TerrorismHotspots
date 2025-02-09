# Imported Libraries
import pandas as pd
import numpy as np
import plotly.express as px
from dash import dash_table

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly import graph_objects as go
import dash_bootstrap_components as dbc


# App starts here
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ------------------------------------------------------------------------
# Importing csv into pandas dataframe
df = pd.read_csv("globalterrorismdb_0718dist.tar.bz2", compression="bz2")
# Narrows the scope of the data from year 2000 to 2017
filtered_years = df[np.logical_and(df['iyear'] >= 2000, df['iyear'] <= 2017)]
# Create a DataFrame with the relevant data
mapdf = pd.DataFrame(filtered_years, columns=['city', 'country_txt', 'region_txt', 'attacktype1_txt', 'latitude', 'longitude', 'success', 'iyear' ])

# Calculate cumulative incidents per country across years for the map
mapdf['cumulative_incidents'] = (
    mapdf.sort_values(by=['country_txt', 'iyear'])  # Sort by country and year to ensure correct order
    .groupby('country_txt')['success']             # Group by country
    .cumsum()                               # Calculate cumulative sum for each country
)
# Aggregate incidents per country per year (sum incidents per year per country)
mapdf_yearly = (
    mapdf.groupby(['country_txt', 'iyear'])['success']  # Group by country and year
    .sum()  # Sum incidents per year per country
    .reset_index()
)

# Compute cumulative incidents per country over years
mapdf_yearly['cumulative_incidents'] = (
    mapdf_yearly
    .sort_values(by=['country_txt', 'iyear'])  # Sort by country and year
    .groupby('country_txt')['success']  # Group by country
    .cumsum()  # Compute cumulative sum per country
)

# Finding the maximum cumulative incidents at the last year (2017) for color scaling
max_incidents = mapdf_yearly[mapdf_yearly['iyear'] == 2017]['cumulative_incidents'].max()

# Figure One -------------------------------------------------
# Figure One: Terror Attacks in Central America & Caribbean (2000-2017)
dff = mapdf.copy() # Create a copy of the mapdf, to not affect the original data
# Filter dataset for the specific region
df_central_america = dff[dff['region_txt'] == 'Central America & Caribbean']

# Group by year and count incidents
df_yearly = df_central_america.groupby('iyear')['cumulative_incidents'].count().reset_index()

# Get incident counts for 2000 and 2017 for the indicator
attacks_2000 = df_yearly[df_yearly['iyear']== 2000]['cumulative_incidents'].values[0]
attacks_2017 = df_yearly[df_yearly['iyear']== 2017]['cumulative_incidents'].values[0]
df_central_america = dff[dff['region_txt'] == 'Central America & Caribbean']

fig_one = go.Figure()

# Add indicator for 2000 vs 2017 comparison
fig_one.add_trace(go.Indicator(
        mode="number+delta",
        value=attacks_2017,
        delta={'reference': attacks_2000, 'valueformat': '0.f'},
        title={'text': 'Terror Attacks in 2000 vs. 2017'},
        domain={'y': [0, 1], 'x': [0.25, 0.75]}
))

# Add line plot for yearly trend
fig_one.add_trace(go.Scatter(
    x=df_yearly['iyear'],
    y=df_yearly['cumulative_incidents'],
    mode='lines+markers',
    name="Terror Attacks"
))

# Update layout
fig_one.update_layout(
    title="Terror Attacks in Central America & Caribbean (2000-2017)",
    xaxis_title="Year",
    yaxis_title="Number of Attacks",
    xaxis={'range': [2000, 2017]}
)

# Figure 2 -------------------------------------------------
# Figure Two: Terror Attacks in Middle East & North Africa (2000-2017)
df_middle_east = dff[dff['region_txt'] == 'Middle East & North Africa']

# Group by year and count incidents
df_yearly_middle_east = df_middle_east.groupby('iyear')['cumulative_incidents'].count().reset_index()

# Get incident counts for 2000 and 2017 for the indicator
attacks_2000_middle_east = df_yearly_middle_east[df_yearly_middle_east['iyear']== 2000]['cumulative_incidents'].values[0]
attacks_2017_middle_east = df_yearly_middle_east[df_yearly_middle_east['iyear']== 2017]['cumulative_incidents'].values[0]
df_middle_east = dff[dff['region_txt'] == 'Middle East & North Africa']

fig_two = go.Figure()

# Add indicator for 2000 vs 2017 comparison
fig_two.add_trace(go.Indicator(
        mode="number+delta",
        value=attacks_2017_middle_east,
        delta={'reference': attacks_2000_middle_east, 'valueformat': '0.f'},
        title={'text': 'Terror Attacks in 2000 vs. 2017'},
        domain={'y': [0, 1], 'x': [0.25, 0.75]}
))

# Add line plot for yearly trend
fig_two.add_trace(go.Scatter(
    x=df_yearly_middle_east['iyear'],
    y=df_yearly_middle_east['cumulative_incidents'],
    mode='lines+markers',
    name="Terror Attacks"
))

# Update layout
fig_two.update_layout(
    title="Terror Attacks in Middle East & North Africa (2000-2017)",
    xaxis_title="Year",
    yaxis_title="Number of Attacks",
    xaxis={'range': [2000, 2017]}
)
# Scatter Geo map -------------------------------------------------------------------------------
# Scatter Geo map: Terrorist Attacks by Country (2000-2017)
dff = mapdf.copy()

# Filter and group data for the years 2000 and 2017
df_2000 = dff[dff["iyear"] == 2000].groupby("region_txt")["cumulative_incidents"].count().reset_index()
df_2017 = dff[dff["iyear"] == 2017].groupby("region_txt")["cumulative_incidents"].count().reset_index()

# Rename columns for clarity in regional growth calculation
df_2000.rename(columns={"cumulative_incidents": "Incidents_2000"}, inplace=True)
df_2017.rename(columns={"cumulative_incidents": "Incidents_2017"}, inplace=True)

# Merge the two DataFrames on 'region_txt'
df_growth = pd.merge(df_2000, df_2017, on="region_txt", how="inner").fillna(0)

# Calculate percentage growth
df_growth["Percentage_Growth"] = ((df_growth["Incidents_2017"] - df_growth["Incidents_2000"]) / df_growth["Incidents_2000"]) * 100

# Replace infinite values (due to division by zero) with NaN
df_growth.replace([float("inf"), float("-inf")], None, inplace=True)
print('This is the percentage growth in terror attacks for each region (2000-2017)\n', df_growth)
    
# Plot the scatter geo map 
fig = px.scatter_geo(
    mapdf_yearly,
    locations='country_txt',
    locationmode="country names",
    color='cumulative_incidents',  # Use cumulative incidents
    hover_name='country_txt',
    size='cumulative_incidents',
    projection="natural earth",
    color_continuous_scale='orrd',
    range_color=[0, max_incidents],
    animation_frame='iyear',  # Add slider functionality
    hover_data={'country_txt': False, 'cumulative_incidents': True}
)

# Update Layout  
fig.update_layout(
    title='Terrorist Attacks by Country (2000-2017)',
    title_x=0.02,  # Left-align the title
    title_font=dict(size=20)  # Optional: Set the title font size
)   

# Update Layout  
fig.update_layout(
    # Add a left-aligned title to the figure
    coloraxis_colorbar_title="Cumulative Incidents",  # Change color bar title
        coloraxis_colorbar=dict(
            title=dict(
                text="Cumulative Incidents",  # Ensure title text is set
                font=dict(
                    color="black",  # Set font color to black
                    size=14,  # Adjust size if needed
                    family="Verdana, sans-serif"  # Set font family
            )
        )
    )
)      

# Update hover template to customize the data shown
fig.update_traces(
        hovertemplate='<b>Country:</b> %{customdata[0]}<br><b>Cumulative Incidents:</b> %{customdata[1]}<extra></extra>',
        textposition='top center'  # Adjust the text position for better readability
)

# Heatmap - use 'region_txt' ------------------------------------------------------------------
filtered_df = pd.DataFrame(filtered_years, columns=['region_txt', 'attacktype1_txt', 'success'])

# Calculate total incidents per region across all years
number_incidents = filtered_df.groupby(['region_txt'])['success'].sum().reset_index()
   
# Calculate total incidents for each region
number_incidents_per_region = number_incidents.groupby(['region_txt'])['success'].sum().reset_index()

# Ensure all 12 regions are considered even if they have zero attacks in a particular year
all_regions = filtered_df['region_txt'].unique()

# Filter the number_incidents DataFrame to include only top 10 countries
number_incidents_per_region = number_incidents_per_region.set_index('region_txt').reindex(all_regions, fill_value=0).reset_index()
# print(all_regions)
# print(number_incidents_per_region)
    
# Heatmap -------------------------------------------------------------
heat_map_two = filtered_df[['region_txt', 'attacktype1_txt', 'success']]
    
# Group by region and attack type, count occurrences across all years
attack_counts = heat_map_two.groupby(['region_txt', 'attacktype1_txt']).size().reset_index(name='attack_count')

# Display the result
print(attack_counts)
# Merge the two dataframes on the country_txt
merged_df = pd.merge(number_incidents_per_region, attack_counts, on="region_txt", how="left")
print(merged_df.columns)
# If a country in top_10 has no matching attack types, it will still be included (with NaN for attack details).
# If a country in top_10 has no recorded attack types, the merge will result in NaN values. You can replace them with 0 like this:
merged_df.fillna(0, inplace=True)
print(merged_df)
    
# Pivot DataFrame to create a matrix format
heatmap_data = merged_df.pivot_table(index='attacktype1_txt', 
                                    columns='region_txt', 
                                    values='attack_count', 
                                    fill_value=0)  # Fill NaNs with 0

# Create heatmap
heat_fig = px.imshow(heatmap_data,
            labels=dict(x="Region", y="Attack Method", color="Attack Method Frequency"), 
            width=1000,   # Adjust width
            height=750,   # Adjust height
            text_auto=True,
            color_continuous_scale='orrd') # Show numbers inside squares 
    
# Adjust layout
heat_fig.update_layout(
    title='Frequency of Terrorist Attack Methods Across Regions (2000-2017)',
    title_x=0.5,  # Left-align the title
    title_font=dict(size=20)  # Optional: Set the title font size
)

# Adjust layout to center heatmap
heat_fig.update_layout(
    xaxis=dict(side="top", automargin=True),
    yaxis=dict(automargin=True),
    margin=dict(l=50, r=50, t=50, b=50),  # Equal margins for centering
    plot_bgcolor="white",  # Background color for clarity
    autosize=True,  # Ensures figure resizes properly in Dash
    template="plotly_white"  # Clean theme
)


# Table -------------------------------------------------
# Define the function early 
def most_effective_attack_method(filtered_years, alpha=10):
    scatter = pd.DataFrame(filtered_years, columns=['city', 'country_txt', 'region_txt', 'nkill', 'nwound', 'success', 'iyear', 'attacktype1_txt'])

    # Calculating casualities, from killed and injured people
    scatter['casualties'] = scatter['nkill'].fillna(0) + scatter['nwound'].fillna(0)
    
    # Calculates the number of casualties and terror attack total incidents per attack type
    attack_stats_one = scatter.groupby('attacktype1_txt').agg(
        casualties=('casualties', 'sum'),
        total_incidents=('success', 'sum')
    ).reset_index()

    # Different Methods used to determine what attack method types were most effective

    # Calculates the ratio of casualties to total incidents for each attack type, and adds a weighting or 10 to normalise extreme values
    attack_stats_one['weighted_effectiveness'] = attack_stats_one['casualties'] / (attack_stats_one['total_incidents'] + alpha)
    
    # Calculates a ratio using logarithmically transformed values of casualties and total incidents
    attack_stats_one['log_effectiveness'] = np.log1p(attack_stats_one['casualties']) / (np.log1p(attack_stats_one['total_incidents'])**2)

    # Calculates a composite score for each attack type using a combination of a simple ratio and a logarithmically transformed value
    attack_stats_one['composite_score'] = (attack_stats_one['casualties'] / attack_stats_one['total_incidents']) * np.log1p(attack_stats_one['total_incidents'])

    # An array of header names 
    static_columns = ['attacktype1_txt', 'weighted_effectiveness', 'composite_score', 'casualties', 'total_incidents', 'log_effectiveness']
    
    # Create a new DataFrame directly from attack_stats_one, selecting only the columns we need
    attack_methods_table = attack_stats_one[static_columns].copy()
    
    return attack_methods_table


# App Layout ------------------------------------------------------------------------

# Create a table of attack methods and their effectiveness metrics
attack_methods_table = most_effective_attack_method(filtered_years)

# Define the main layout of the Dash application
app.layout = html.Div([
    # Main title of the dashboard
    html.H1('Analyzing Global Terrorism: Targets, Methods, and Impacts', style={'font-family':'Verdana, sans-serif', 'text-align':'center', 'font-weight': 'normal'}),
    
    html.Br(), # Add a line break for spacing

    # World map showing number of incidents
    dcc.Graph(id='num_incidents_map', figure=fig),
    
    # Row containing two side-by-side graphs
    dbc.Row([
        # Graph for Central America terror attacks
        dbc.Col(dcc.Graph(
            id='central-america-terror-attacks',
            figure=fig_one
        ), width=6),  # Takes half the width of the row
        
        # Graph for Middle East terror attacks
        dbc.Col(dcc.Graph(
            id='middle-east-terror-attacks',
            figure=fig_two
        ), width=6)  # Takes half the width of the row
    ]),
    
    # Heat map graph, centered on the page
    html.Div([
    dcc.Graph(id='heat_map_graph', figure=heat_fig)
], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
   
    # Dropdown menu for selecting attack type
    html.Div(
        dcc.Dropdown(
            options=[{'label': val, 'value': val} for val in mapdf['attacktype1_txt'].unique()],
            value='Armed Assault',
            id='dropdown'   
        ),
        style={
            'width':'50%', # Dropdown takes half of the page width
            'margin-left': '25px' # Adds left padding
        }
    ),
   
    html.Br(),# Add a line break for spacing
    # Scatter plot showing correlation
    html.Div(
        dcc.Graph(id='scatter_correlation', figure={}),
        style = {
            'width': '80%',  # Make graph take 80% of the screen width
            'margin': 'auto',  # Center the graph horizontally
            'padding': '20px'  # Add padding around it
        }
    ),
    html.Br(), # Add a line break for spacing
    
    html.Div([
    # Title for the table
    html.H5('Most Effective Terrorism Attack Methods', 
            style={'text-align': 'center', 
                   'font-family': 'Verdana, sans-serif', 
                   'font-weight': 'normal'}),
    
    # DataTable component
    dash_table.DataTable(
        id='attack-table',
        columns=[
            {'name': 'Attack Method', 'id': 'attacktype1_txt'},  
            {'name': 'Weighted Effectiveness', 'id': 'weighted_effectiveness'},  
            {'name': 'Composite Score', 'id': 'composite_score'},  
            {'name': 'Casualties', 'id': 'casualties'},
            {'name': 'Total Incidents', 'id': 'total_incidents'},
            {'name': 'Log Effectiveness', 'id': 'log_effectiveness'}
        ],
        data=attack_methods_table.to_dict('records'),
        # Conditional styling for cells above median value
        style_data_conditional=[
            {
                'if': {'column_id': col, 'filter_query': f'{{{col}}} > {attack_methods_table[col].median()}'},
                'backgroundColor': 'tomato'
            } for col in attack_methods_table.columns[1:]  # Apply to numeric columns only
        ],
        style_table={'overflowX': 'auto'}, # Allow horizontal scrolling if needed
        style_header={'backgroundColor': 'transparent', 'color': 'black', 'fontWeight': 'bold', 'textAlign': 'left', 'font-family': 'Verdana, sans-serif'},
        style_cell={'textAlign': 'left', 'font-family': 'Verdana, sans-serif'} 
    ),
], style={
    'width': '80%',  # Make table take 80% of the screen width
    'margin': 'auto',  # Center it horizontally
    'padding': '20px'  # Add padding
})

])
# ------------------------------------------------------------------------
# Connect the Plotly graphs with Dash components
@app.callback(
     Output(component_id = 'scatter_correlation', component_property='figure'),
    [Input(component_id='dropdown', component_property='value')]
) 
def update_scatter(dropdown_value):
    '''
    Updates the scatter plot based on the selected attack type from the dropdown.

    Args:
        dropdown_value (str): The attack type selected from the dropdown.

    Returns:
        plotly.graph_objects.Figure: Updated scatter plot figure.
    '''
    
    print(dropdown_value)
    print(type(dropdown_value))  # Debugging: check the value and type from the slider

    # Ensure `filtered_years` is defined 
    scatter = pd.DataFrame(filtered_years, columns=['city', 'country_txt', 'region_txt', 'nkill', 'nwound', 'success', 'iyear', 'attacktype1_txt'])

    country_to_region = scatter.copy()
    country_to_region = country_to_region.drop(['city','nkill', 'nwound', 'success', 'iyear', 'attacktype1_txt'], axis=1)
    
    # Filter the scatter dataset
    filtered_scatter = scatter[scatter['attacktype1_txt'] == dropdown_value]
    # Calculate casualties and total incidents for the filtered data
    filtered_scatter['casualties'] = filtered_scatter['nkill'].fillna(0) + filtered_scatter['nwound'].fillna(0)

    
    # Aggregate data by country
    result = filtered_scatter.groupby('country_txt').agg({
        'success': 'sum',
        'casualties': 'sum'
    }).reset_index()

    result.rename(columns={'success': 'total_incidents'}, inplace=True)

    print(f"This code describes the data for {dropdown_value}\n", result.describe())

    # Calculate Correlation
    # What is the relationship between the number of incidents, and casualties, based on the attack method used
    correlation = result['total_incidents'].corr(result['casualties'])
    print('Correlation coefficient: ', correlation)

    # Create the mapping of country to region by selecting unique values
    country_dictionary = country_to_region[['country_txt', 'region_txt']].drop_duplicates().set_index('country_txt').to_dict()['region_txt']
    # Check the first few entries of the dictionary 
    print("Sample country-to-region mapping:\n", list(country_dictionary.items())[:10])
    # Create a new column in the result DataFrame that maps country to region
    result['Region'] = result['country_txt'].map(country_dictionary)
    # Create the scatter plot
    scatter_chart = px.scatter(
        result,
        x='total_incidents',
        y='casualties',
        color='Region',
        title=f"Number of Incidents vs Number of Casualties for {dropdown_value}",
        labels={'total_incidents': 'Number of Incidents', 'casualties': 'Number of Casualties'},
        hover_data=['country_txt']
    )

    # Add subheading as an annotation
    scatter_chart.add_annotation(
    xref="paper", yref="paper",  # Reference to the whole figure
    x=0.5, y=1.05,  # Position (slightly above title)
    text= f"Correlation coefficient: {correlation}",  # Subheading text
    showarrow=False,
    font=dict(size=14)  # Adjust font size
    )
    
    return scatter_chart

# ------------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)

