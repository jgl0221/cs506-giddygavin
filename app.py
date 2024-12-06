import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load datasets
data_2021 = pd.read_csv("world-happiness-report-2021.csv")
data_2022 = pd.read_csv("World Happiness Report 2022.csv")
data_2023 = pd.read_csv("2023.csv")

# Standardize column names for all datasets
data_2021.columns = data_2021.columns.str.strip().str.replace(" ", "_").str.lower()
data_2022.columns = data_2022.columns.str.strip().str.replace(" ", "_").str.lower()
data_2023.columns = data_2023.columns.str.strip().str.replace(" ", "_").str.lower()

# Rename columns in 2022 dataset for consistency
data_2022.rename(columns={
    "country": "country_name", 
    "happiness_score": "ladder_score"  # Align with 2021 and 2023
}, inplace=True)

# Debugging: Print column names to verify consistency
print("2021 Columns:", data_2021.columns)
print("2022 Columns:", data_2022.columns)
print("2023 Columns:", data_2023.columns)

# Add a 'year' column to each dataset
data_2021['year'] = 2021
data_2022['year'] = 2022
data_2023['year'] = 2023

# Combine datasets into a single DataFrame
data = pd.concat([data_2021, data_2022, data_2023], ignore_index=True)

# Debugging: Print combined DataFrame columns
print("Combined DataFrame Columns:", data.columns)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("World Happiness Statistics (2021–2023)", style={'textAlign': 'center'}),
    dcc.Input(
        id="year-input",
        type="number",
        placeholder="Enter Year (2021–2023)",
        min=2021,
        max=2023,
        step=1,
        value=2021,
        style={'marginBottom': '20px', 'width': '50%', 'margin': '0 auto'}
    ),
    html.Div(id="output-message", style={'textAlign': 'center'}),
    dcc.Graph(id="happiness-graph")
])

@app.callback(
    [Output("output-message", "children"),
     Output("happiness-graph", "figure")],
    [Input("year-input", "value")]
)
def update_graph(year):
    if year not in [2021, 2022, 2023]:
        return "Please enter a valid year (2021–2023).", dash.no_update

    # Filter data for the selected year
    filtered_data = data[data["year"] == year]
    print(f"Filtered Data for {year}:")
    print(filtered_data.head())  # Debugging statement

    if filtered_data.empty:
        return f"No data available for {year}.", dash.no_update

    # Create a bar chart
    fig = px.bar(
        filtered_data,
        x="country_name",
        y="ladder_score",  # Standardized column name
        title=f'Happiness Scores for {year}',
        labels={"ladder_score": "Happiness Score", "country_name": "Country"},
        text="ladder_score"
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis={'categoryorder': 'total descending'})

    return f"Displaying data for {year}.", fig

if __name__ == "__main__":
    app.run_server(debug=True)
