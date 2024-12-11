import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define features with the updated normalized naming
features = [
    'normalized_gdp_per_capita',
    'social_support',
    'normalized_healthy_life_expectancy',
    'freedom_to_make_life_choices',
    'generosity',
    'corruption'
]

# Load and process the datasets
file_2018 = pd.read_csv('2018.csv')
file_2019 = pd.read_csv('2019.csv')
file_2020 = pd.read_csv('2020.csv')
file_2021 = pd.read_csv('2021.csv')
file_2022 = pd.read_csv('2022.csv')
processed_2023 = pd.read_csv('processed_2023.csv')

# Define the common columns to keep across datasets
common_columns = [
    'country_name', 'year', 'life_ladder', 'normalized_gdp_per_capita',
    'social_support', 'normalized_healthy_life_expectancy',
    'freedom_to_make_life_choices', 'generosity', 'corruption'
]

# Rename columns in the historical files to use normalized naming
file_2018.rename(columns={
    'Country or region': 'country_name',
    'Score': 'life_ladder',
    'GDP per capita': 'normalized_gdp_per_capita',
    'Social support': 'social_support',
    'Healthy life expectancy': 'normalized_healthy_life_expectancy',
    'Freedom to make life choices': 'freedom_to_make_life_choices',
    'Generosity': 'generosity',
    'Perceptions of corruption': 'corruption'
}, inplace=True)

file_2019.rename(columns={
    'Country or region': 'country_name',
    'Score': 'life_ladder',
    'GDP per capita': 'normalized_gdp_per_capita',
    'Social support': 'social_support',
    'Healthy life expectancy': 'normalized_healthy_life_expectancy',
    'Freedom to make life choices': 'freedom_to_make_life_choices',
    'Generosity': 'generosity',
    'Perceptions of corruption': 'corruption'
}, inplace=True)

file_2020.rename(columns={
    'Country name': 'country_name',
    'Ladder score': 'life_ladder',
    'Logged GDP per capita': 'normalized_gdp_per_capita',
    'Social support': 'social_support',
    'Healthy life expectancy': 'normalized_healthy_life_expectancy',
    'Freedom to make life choices': 'freedom_to_make_life_choices',
    'Generosity': 'generosity',
    'Perceptions of corruption': 'corruption'
}, inplace=True)

file_2021.rename(columns={
    'Country name': 'country_name',
    'Ladder score': 'life_ladder',
    'Logged GDP per capita': 'normalized_gdp_per_capita',
    'Social support': 'social_support',
    'Healthy life expectancy': 'normalized_healthy_life_expectancy',
    'Freedom to make life choices': 'freedom_to_make_life_choices',
    'Generosity': 'generosity',
    'Perceptions of corruption': 'corruption'
}, inplace=True)

file_2022.rename(columns={
    'Country': 'country_name',
    'Happiness score': 'life_ladder',
    'Explained by: GDP per capita': 'normalized_gdp_per_capita',
    'Explained by: Social support': 'social_support',
    'Explained by: Healthy life expectancy': 'normalized_healthy_life_expectancy',
    'Explained by: Freedom to make life choices': 'freedom_to_make_life_choices',
    'Explained by: Generosity': 'generosity',
    'Explained by: Perceptions of corruption': 'corruption'
}, inplace=True)

# Add 'year' column to each dataset
file_2018['year'] = 2018
file_2019['year'] = 2019
file_2020['year'] = 2020
file_2021['year'] = 2021
file_2022['year'] = 2022

# Keep only the common columns
file_2018 = file_2018[common_columns]
file_2019 = file_2019[common_columns]
file_2020 = file_2020[common_columns]
file_2021 = file_2021[common_columns]
file_2022 = file_2022[common_columns]

# Process 2023 data
processed_2023.rename(columns={
    'Country': 'country_name',
    'Happiness score': 'life_ladder',
    'Explained by: Social support': 'social_support',
    'Explained by: Healthy life expectancy': 'normalized_healthy_life_expectancy',
    'Explained by: Freedom to make life choices': 'freedom_to_make_life_choices',
    'Explained by: Generosity': 'generosity',
    'Explained by: Perceptions of corruption': 'corruption',
    'Explained by: GDP per capita': 'normalized_gdp_per_capita'
}, inplace=True)

# Ensure all features are present in processed_2023
for col in features:
    if col not in processed_2023.columns:
        processed_2023[col] = 0

processed_2023['year'] = 2023
processed_2023 = processed_2023[common_columns]

# Combine all years of data (2018-2022) for training
combined_data = pd.concat([file_2018, file_2019, file_2020, file_2021, file_2022], ignore_index=True)
combined_data.dropna(inplace=True)

dict_df = {
    "2018": file_2018.set_index("country_name"),
    "2019": file_2019.set_index("country_name"),
    "2020": file_2020.set_index("country_name"),
    "2021": file_2021.set_index("country_name"),
    "2022": file_2022.set_index("country_name")
}

# Train the model
X = combined_data[features]
y = combined_data['life_ladder']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# Pre-scale 2023 data for comparison
X_2023 = processed_2023[features]
X_2023_scaled = scaler.transform(X_2023)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("World Happiness Report (2018-2022)", style={'textAlign': 'center'}),
    html.Label("Select Year:", style={'fontSize': '20px', 'color': '#333', 'marginBottom': '10px'}),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': year, 'value': year} for year in combined_data['year'].unique()],
        value=2018,
        style={'width': '50%', 'margin': '0 auto', 'fontSize': '16px'}
    ),
    dcc.Graph(id='ladder-score-graph'),
    dcc.Graph(id='map-visualization'),
    dcc.Graph(id='feature-statistics-graph'),
    html.Hr(),
    html.H3("How Happiness Changed", style={'textAlign': 'center', 'fontSize': '24px', 'color': '#333'}),
    html.Label("Select Year Range:", style={'fontSize': '20px', 'color': '#333'}),
    html.Div([
        dcc.Dropdown(
            id='change-year-from',
            options=[{'label': year, 'value': year} for year in dict_df.keys()],
            value='2018',
            style={'width': '45%', 'display': 'inline-block', 'marginRight': '5%'}
        ),
        dcc.Dropdown(
            id='change-year-to',
            options=[{'label': year, 'value': year} for year in dict_df.keys()],
            value='2021',
            style={'width': '45%', 'display': 'inline-block'}
        ),
    ]),
    dcc.Graph(id='happiness-change-graph'),
    html.Hr(),
    html.H3("Predicting Happiness Score for 2023", style={'textAlign': 'center', 'fontSize': '24px', 'color': '#333'}),

    # Sliders for the input features with updated labels
    html.Div([
        html.Label('Normalized GDP per Capita:'),
        dcc.Slider(
            id='input-normalized_gdp_per_capita',
            min=0.80, max=1.00, step=0.01, value=0.85,
            marks={i: f'{i:.2f}' for i in np.arange(0.80, 1.01, 0.05)}
        ),
        html.Label('Social Support:'),
        dcc.Slider(
            id='input-social_support',
            min=0.85, max=1.00, step=0.01, value=0.90,
            marks={i: f'{i:.2f}' for i in np.arange(0.85, 1.01, 0.05)}
        ),
        html.Label('Normalized Healthy Life Expectancy:'),
        dcc.Slider(
            id='input-normalized_healthy_life_expectancy',
            min=0.70, max=0.85, step=0.001, value=0.78,
            marks={i: f'{i:.2f}' for i in np.arange(0.70, 0.851, 0.05)}
        ),
        html.Label('Freedom to Make Life Choices:'),
        dcc.Slider(
            id='input-freedom_to_make_life_choices',
            min=0.80, max=1.00, step=0.01, value=0.90,
            marks={i: f'{i:.2f}' for i in np.arange(0.80, 1.01, 0.05)}
        ),
        html.Label('Generosity:'),
        dcc.Slider(
            id='input-generosity',
            min=-0.05, max=0.25, step=0.01, value=0.05,
            marks={i: f'{i:.2f}' for i in np.arange(-0.05, 0.26, 0.05)}
        ),
        html.Label('Corruption (Perceptions):'),
        dcc.Slider(
            id='input-corruption',
            min=0.15, max=0.75, step=0.01, value=0.20,
            marks={i: f'{i:.2f}' for i in np.arange(0.15, 0.76, 0.15)}
        ),
    ], style={'padding': '20px'}),

    html.Div(id='prediction-results', style={'padding': '20px', 'fontSize': '16px'}),
    dcc.Graph(id='prediction-graph')
])

@app.callback(
    Output('happiness-change-graph', 'figure'),
    [Input('change-year-from', 'value'), Input('change-year-to', 'value')]
)
def update_happiness_change(year_from, year_to):
    yfrom = dict_df[year_from]
    yto = dict_df[year_to]
    
    # Align and calculate change
    common_countries = yfrom.index.intersection(yto.index)
    yfrom = yfrom.loc[common_countries]
    yto = yto.loc[common_countries]
    
    df_change = yfrom.copy()
    df_change['Change'] = (yto['life_ladder'] - yfrom['life_ladder']) / yfrom['life_ladder']
    df_change = df_change.sort_values('Change').dropna()

    fig = px.bar(
        df_change.reset_index(),
        x='Change',
        y='country_name',
        orientation='h',
        title=f"How happiness changed from {year_from} to {year_to}",
        color='Change',
        color_continuous_scale='RdYlGn',
        height=800
    )
    return fig

@app.callback(
    [Output('ladder-score-graph', 'figure'),
     Output('map-visualization', 'figure'),
     Output('feature-statistics-graph', 'figure')],
    [Input('year-dropdown', 'value')]
)
def update_graphs(selected_year):
    filtered_data = combined_data[combined_data['year'] == selected_year]

    # Ladder Score Bar Chart
    ladder_fig = px.bar(
        filtered_data,
        x='country_name',
        y='life_ladder',
        title=f"Ladder Score by Country for {selected_year}",
        labels={'life_ladder': 'Ladder Score', 'country_name': 'Country'}
    )

    # Map Visualization
    map_fig = px.choropleth(
        filtered_data,
        locations='country_name',
        locationmode='country names',
        color='life_ladder',
        title=f"Happiness Map for {selected_year}",
        color_continuous_scale='Viridis',
        labels={'life_ladder': 'Happiness Score'}
    )
    map_fig.update_layout(geo=dict(showframe=False, projection_type='equirectangular'))

    # Feature Statistics
    stats = {
        'Feature': features,
        'Average': [filtered_data[feature].mean() for feature in features],
        'Max': [filtered_data[feature].max() for feature in features],
        'Max Country': [
            filtered_data.loc[filtered_data[feature].idxmax(), 'country_name']
            for feature in features
        ]
    }

    stats_df = pd.DataFrame(stats)

    stats_fig = px.bar(
        stats_df,
        x='Feature',
        y=['Average', 'Max'],
        barmode='group',
        title=f"Feature Statistics for {selected_year}",
        labels={'value': 'Score', 'Feature': 'Feature'},
        hover_data={'Max Country': True}
    )

    return ladder_fig, map_fig, stats_fig

@app.callback(
    [Output('prediction-results', 'children'),
     Output('prediction-graph', 'figure')],
    [Input('input-normalized_gdp_per_capita', 'value'),
     Input('input-social_support', 'value'),
     Input('input-normalized_healthy_life_expectancy', 'value'),
     Input('input-freedom_to_make_life_choices', 'value'),
     Input('input-generosity', 'value'),
     Input('input-corruption', 'value')]
)
def predict_happiness_2023(gdp, social, healthy, freedom, generosity, corruption):
    # Prepare input feature array
    input_features = np.array([[gdp, social, healthy, freedom, generosity, corruption]])
    input_scaled = scaler.transform(input_features)
    predicted_score = model.predict(input_scaled)[0]

    # Find the closest country in 2023 data
    distances = np.linalg.norm(X_2023_scaled - input_scaled, axis=1)
    closest_idx = np.argmin(distances)
    closest_country = processed_2023.iloc[closest_idx]
    closest_country_name = closest_country['country_name']
    actual_score = closest_country['life_ladder']

    results_text = html.Div(style={'lineHeight': '1.6', 'fontSize': '18px'}, children=[
        html.P(f"Predicted Happiness Score for 2023 (given inputs): {predicted_score:.3f}"),
        html.P(f"Most Similar Country in 2023: {closest_country_name}"),
        html.P(f"Actual Happiness Score of {closest_country_name}: {actual_score:.3f}")
    ])

    # Comparison figure
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Predicted Score', f'Actual Score ({closest_country_name})'],
        y=[predicted_score, actual_score],
        marker_color=['blue', 'orange']
    ))
    fig.update_layout(
        title="Predicting Happiness Score for 2023",
        yaxis_title="Happiness Score"
    )

    return results_text, fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
