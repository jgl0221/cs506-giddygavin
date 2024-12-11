import pytest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Test for loading the data
def test_load_data():
    # Load the dataset for 2018
    data = pd.read_csv('Combined - 2018.csv')
    assert data is not None
    assert 'Country or region' in data.columns  # Adjusted column name based on dataset
    assert 'Score' in data.columns  # Adjusted column name based on dataset

# Test for dataset processing (renaming columns, filtering, etc.)
def test_process_dataset():
    # Load and process the dataset for 2018
    data = pd.read_csv('Combined - 2018.csv')
    
    # Apply the renaming and filtering logic
    data.rename(columns={
        'Country or region': 'country_name',
        'Score': 'life_ladder',
        'Social support': 'social_support',
        'Healthy life expectancy': 'healthy_life_expectancy_at_birth',
        'Freedom to make life choices': 'freedom_to_make_life_choices',
        'Generosity': 'generosity',
        'Perceptions of corruption': 'corruption'
    }, inplace=True)
    
    # Filter to keep only relevant columns
    common_columns = ['country_name', 'life_ladder', 'social_support', 
                      'healthy_life_expectancy_at_birth', 'freedom_to_make_life_choices', 
                      'generosity', 'corruption']
    processed_data = data[common_columns]
    
    # Check that all required columns exist
    expected_columns = ['country_name', 'life_ladder', 'social_support', 
                        'healthy_life_expectancy_at_birth', 'freedom_to_make_life_choices', 
                        'generosity', 'corruption']
    assert all(col in processed_data.columns for col in expected_columns)

# Test for normalization of GDP and life expectancy
def test_normalization():
    data = pd.read_csv('Combined - 2018.csv')
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalize GDP per capita
    data['normalized_gdp_per_capita'] = scaler.fit_transform(data['GDP per capita'].values.reshape(-1, 1))
    assert data['normalized_gdp_per_capita'].min() >= 0
    assert data['normalized_gdp_per_capita'].max() <= 1

    # Normalize healthy life expectancy
    data['normalized_healthy_life_expectancy'] = scaler.fit_transform(data['healthy_life_expectancy_at_birth'].values.reshape(-1, 1))
    assert data['normalized_healthy_life_expectancy'].min() >= 0
    assert data['normalized_healthy_life_expectancy'].max() <= 1

# Test for combined data processing (merging data for 2018-2022)
def test_combined_data_processing():
    # Load all the data files (2018-2022)
    dataframes = {}
    for year in range(2018, 2023):
        dataframes[year] = pd.read_csv(f'Combined - {year}.csv')

    # Combine the datasets for 2018-2022
    combined_data_2018_2022 = pd.concat(
        [dataframes[year] for year in range(2018, 2023)], ignore_index=True
    )

    # Check that the combined dataset has the correct number of rows (760 as per your case)
    assert combined_data_2018_2022.shape[0] == 760  # Adjust to the correct row count

    # Ensure necessary columns are present
    expected_columns = ['country_name', 'life_ladder', 'social_support', 
                        'healthy_life_expectancy_at_birth', 'freedom_to_make_life_choices', 
                        'generosity', 'corruption', 'year', 'normalized_healthy_life_expectancy']
    assert all(col in combined_data_2018_2022.columns for col in expected_columns)

# Test for training and evaluating a Random Forest model
def test_model_training():
    # Assuming combined_data_2018_2022 and processed_dataframes for 2023 exist
    combined_data_2018_2022 = pd.read_csv('combined_data_2018_2022.csv')
    processed_data_2023 = pd.read_csv('processed_2023.csv')

    # Features and target variable
    features = [
        'social_support', 'freedom_to_make_life_choices', 'generosity',
        'corruption', 'normalized_gdp_per_capita', 'normalized_healthy_life_expectancy'
    ]
    target = 'life_ladder'

    # Split the data for training (2018-2022) and testing (2023)
    X_train = combined_data_2018_2022[features].dropna()
    y_train = combined_data_2018_2022[target].loc[X_train.index]
    X_test = processed_data_2023[features].dropna()
    y_test = processed_data_2023[target].loc[X_test.index]

    # Train Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Check if the prediction and true value arrays have the same shape
    assert y_pred.shape == y_test.shape

    # Evaluate the model's performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Remove 'squared' issue by using RMSE directly
    r2 = r2_score(y_test, y_pred)

    # Check that performance metrics are reasonable
    assert mae < 1.0  # Adjust this threshold as needed
    assert mse < 1.0  # Adjust this threshold as needed
    assert r2 >= 0.5  # R² should be reasonable

# Test for model feature importance
def test_feature_importance():
    # Load the combined dataset (2018-2022) and process it
    combined_data_2018_2022 = pd.read_csv('combined_data_2018_2022.csv')

    # Prepare the feature matrix and target vector
    features = [
        'social_support', 'freedom_to_make_life_choices', 'generosity',
        'corruption', 'normalized_gdp_per_capita', 'normalized_healthy_life_expectancy'
    ]
    target = 'life_ladder'
    
    X_train = combined_data_2018_2022[features].dropna()
    y_train = combined_data_2018_2022[target].loc[X_train.index]

    # Train a Random Forest model to evaluate feature importance
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Get feature importances
    importances = rf_model.feature_importances_

    # Ensure feature importance is calculated and has a reasonable sum
    assert sum(importances) > 0
    assert len(importances) == len(features)
