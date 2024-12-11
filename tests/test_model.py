import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pytest

# Test for loading data
def test_load_data():
    data = pd.read_csv('Combined - 2018.csv')
    
    # Ensure that the 'country_name' column exists
    print(data.columns)  # For debugging purposes
    assert data is not None
    assert 'country_name' in data.columns  # Check for 'country_name'

# Test for normalization
def test_normalization():
    data = pd.read_csv('Combined - 2018.csv')
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalize GDP per capita
    data['normalized_gdp_per_capita'] = scaler.fit_transform(data['GDP per capita'].values.reshape(-1, 1))
    assert data['normalized_gdp_per_capita'].min() >= 0
    assert data['normalized_gdp_per_capita'].max() <= 1

    # Check if 'healthy_life_expectancy_at_birth' exists
    if 'healthy_life_expectancy_at_birth' in data.columns:
        data['normalized_healthy_life_expectancy'] = scaler.fit_transform(
            data['healthy_life_expectancy_at_birth'].values.reshape(-1, 1)
        )
        assert data['normalized_healthy_life_expectancy'].min() >= 0
        assert data['normalized_healthy_life_expectancy'].max() <= 1
    else:
        print("Warning: 'healthy_life_expectancy_at_birth' column is missing in the dataset.")

# Test for combined data processing
def test_combined_data_processing():
    # Load all the data files (2018-2022)
    dataframes = {}
    for year in range(2018, 2023):
        dataframes[year] = pd.read_csv(f'Combined - {year}.csv')

    # Combine the datasets for 2018-2022
    combined_data_2018_2022 = pd.concat(
        [dataframes[year] for year in range(2018, 2023)], ignore_index=True
    )

    # Check that the combined dataset has the correct number of rows (adjust as needed)
    assert combined_data_2018_2022.shape[0] == 635  # Adjust based on your dataset

    # Ensure necessary columns are present
    expected_columns = ['country_name', 'life_ladder', 'social_support',
                        'healthy_life_expectancy_at_birth', 'freedom_to_make_life_choices',
                        'generosity', 'corruption', 'year', 'normalized_healthy_life_expectancy']
    
    # Check if all expected columns are in the dataset
    assert all(col in combined_data_2018_2022.columns for col in expected_columns)

# Test for model training
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
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Correct use of RMSE calculation
    r2 = r2_score(y_test, y_pred)

    # Print model performance metrics (optional for debugging)
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")

    assert mae >= 0  # Ensure the MAE is non-negative
    assert mse >= 0  # Ensure the MSE is non-negative
    assert rmse >= 0  # Ensure the RMSE is non-negative
    assert r2 >= 0  # Ensure the R2 score is reasonable

# Run all tests
if __name__ == "__main__":
    pytest.main()
