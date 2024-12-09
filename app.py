import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Load and process the dataset
file_2018 = pd.read_csv('2018.csv')
file_2019 = pd.read_csv('2019.csv')
file_2020 = pd.read_csv('2020.csv')
file_2021 = pd.read_csv('2021.csv')
file_2022 = pd.read_csv('2022.csv')

# Define common columns and rename for consistency
common_columns = [
    'country_name', 'year', 'life_ladder', 'log_gdp_per_capita',
    'social_support', 'healthy_life_expectancy_at_birth',
    'freedom_to_make_life_choices', 'generosity', 'corruption'
]

file_2018.rename(columns={
    'Country or region': 'country_name',
    'Score': 'life_ladder',
    'GDP per capita': 'log_gdp_per_capita',
    'Social support': 'social_support',
    'Healthy life expectancy': 'healthy_life_expectancy_at_birth',
    'Freedom to make life choices': 'freedom_to_make_life_choices',
    'Generosity': 'generosity',
    'Perceptions of corruption': 'corruption'
}, inplace=True)

file_2019.rename(columns={
    'Country or region': 'country_name',
    'Score': 'life_ladder',
    'GDP per capita': 'log_gdp_per_capita',
    'Social support': 'social_support',
    'Healthy life expectancy': 'healthy_life_expectancy_at_birth',
    'Freedom to make life choices': 'freedom_to_make_life_choices',
    'Generosity': 'generosity',
    'Perceptions of corruption': 'corruption'
}, inplace=True)

file_2020.rename(columns={
    'Country name': 'country_name',
    'Ladder score': 'life_ladder',
    'Logged GDP per capita': 'log_gdp_per_capita',
    'Social support': 'social_support',
    'Healthy life expectancy': 'healthy_life_expectancy_at_birth',
    'Freedom to make life choices': 'freedom_to_make_life_choices',
    'Generosity': 'generosity',
    'Perceptions of corruption': 'corruption'
}, inplace=True)

file_2021.rename(columns={
    'Country name': 'country_name',
    'Ladder score': 'life_ladder',
    'Logged GDP per capita': 'log_gdp_per_capita',
    'Social support': 'social_support',
    'Healthy life expectancy': 'healthy_life_expectancy_at_birth',
    'Freedom to make life choices': 'freedom_to_make_life_choices',
    'Generosity': 'generosity',
    'Perceptions of corruption': 'corruption'
}, inplace=True)

file_2022.rename(columns={
    'Country': 'country_name',
    'Happiness score': 'life_ladder',
    'Explained by: GDP per capita': 'log_gdp_per_capita',
    'Explained by: Social support': 'social_support',
    'Explained by: Healthy life expectancy': 'healthy_life_expectancy_at_birth',
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

# Retain only the common columns for each dataset
file_2018 = file_2018[common_columns]
file_2019 = file_2019[common_columns]
file_2020 = file_2020[common_columns]
file_2021 = file_2021[common_columns]
file_2022 = file_2022[common_columns]

# Combine datasets
combined_data = pd.concat([file_2018, file_2019, file_2020, file_2021, file_2022], ignore_index=True)
combined_data.dropna(inplace=True)

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("World Happiness Report (2018-2022)", style={'textAlign': 'center'}),
    html.Label("Select Year:"),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': year, 'value': year} for year in combined_data['year'].unique()],
        value=2018
    ),
    dcc.Graph(id='ladder-score-graph'),
    dcc.Graph(id='feature-statistics-graph')
])

@app.callback(
    [Output('ladder-score-graph', 'figure'),
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

    # Feature Statistics
    features = [
        'log_gdp_per_capita', 'social_support', 'healthy_life_expectancy_at_birth',
        'freedom_to_make_life_choices', 'generosity', 'corruption'
    ]

    stats = {
        'Feature': features,
        'Average': [filtered_data[feature].mean() for feature in features],
        'Max': [filtered_data[feature].max() for feature in features],
        'Max Country': [
            filtered_data.loc[filtered_data[feature].idxmax(), 'country_name']
            for feature in features
        ],
        'Min': [filtered_data[feature].min() for feature in features],
        'Min Country': [
            filtered_data.loc[filtered_data[feature].idxmin(), 'country_name']
            for feature in features
        ]
    }

    stats_df = pd.DataFrame(stats)

    stats_fig = px.bar(
        stats_df,
        x='Feature',
        y=['Average', 'Max', 'Min'],
        barmode='group',
        title=f"Feature Statistics for {selected_year}",
        labels={'value': 'Score', 'Feature': 'Feature'},
        hover_data={
            'Max Country': True,
            'Min Country': True
        }
    ) 

    return ladder_fig, stats_fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)  # Change port to 8051
