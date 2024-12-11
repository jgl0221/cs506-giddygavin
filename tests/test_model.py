import pytest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from your_module import load_data, process_dataset, process_and_scale_data, train_model  # Replace with actual imports


# Test for loading the data
def test_load_data():
    # Assuming you have a load_data function that loads your CSVs
    data = load_data('Combined - 2018.csv')
    assert data is not None
    assert 'country_name' in data.columns
    assert 'life_ladder' in data.columns


# Test for dataset processing (renaming columns, filtering, etc.)
def test_process_dataset():
    # Sample dataset from 2018
    data = pd.read_csv('Combined - 2018.csv')  # Sample data
    processed_data = process_dataset(data, 2018)
    
    # Check that the necessary columns are present
    expected_columns = [
        'country_name', 'life_ladder', 'social_support', 'healthy_life_expectancy_at_birth',
        'freedom_to_make_life_choices', 'generosity', 'corruption', 'year', 'normalized_healthy_life_expectancy'
    ]
    
    assert all(col in processed_data.columns for col in expected_columns)
    assert 'year' in processed_data.columns  # Ensure year column is added
    assert processed_data['year'].iloc[0] == 2018  # Ensure the year column has the correct year


# Test for normalization of GDP and life expectancy
def test_normalization():
    data = pd.read_csv('Combined - 2018.csv')
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Check that the data is scaled within the expected range for GDP
    data['normalized_gdp_per_capita'] = scaler.fit_transform(data['GDP per capita'].values.reshape(-1, 1))
    assert data['normalized_gdp_per_capita'].min() >= 0
    assert data['normalized_gdp_per_capita'].max() <= 1

    # Check that the life expectancy is normalized
    data['normalized_healthy_life_expectancy'] = scaler.fit_transform(data['healthy_life_expectancy'].values.reshape(-1, 1))
    assert data['normalized_healthy_life_expectancy'].min() >= 0
    assert data['normalized_healthy_life_expectancy'].max() <= 1


# Test for correct data processing and model training
def test_combined_data_processing():
    # Process the combined data
    processed_data = pd.read_csv('combined_data_2018_2022.csv')

    # Ensure that after processing, the shape is as expected (635 rows as per your example)
    assert processed_data.shape[0] == 635

    # Ensure that necessary columns exist after processing
    assert 'country_name' in processed_data.columns
    assert 'life_ladder' in processed_data.columns

    # Train a simple model (e.g., Random Forest) using the processed data
    X = processed_data[['social_support', 'freedom_to_make_life_choices', 'generosity', 'corruption', 'normalized_gdp_per_capita', 'normalized_healthy_life_expectancy']]
    y = processed_data['life_ladder']
    
    model = train_model(X, y)
    assert model is not None


# Test for model evaluation metrics
def test_model_evaluation():
    # Assuming train_model returns a trained model
    model = train_model(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Check that the model returns predictions with the correct shape
    assert y_pred.shape == y_test.shape

    # Evaluate the model's performance using R2 score
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    assert r2 >= 0  # R² should be non-negative
