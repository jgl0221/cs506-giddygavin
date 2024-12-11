# Final Writeup: Predicting Happiness Scores with the World Happiness Report
**Josh Lee, Jimin Park, Gavin Park**  
*(Team Giddy Gavin)*

---

## Core Objective:
The objective of this project is to predict Happiness Scores (Life Ladder) using machine learning techniques. Our model utilizes several key explanatory variables that capture critical socioeconomic and well-being factors affecting life satisfaction. These factors include:

- Normalized GDP per capita
- Social support
- Normalized healthy life expectancy at birth
- Freedom to make life choices
- Generosity
- Corruption

We train our model using data from the 2018–2022 World Happiness Reports and evaluate its performance using the 2023 report. This approach aims to measure the predictive accuracy and generalizability of the model on new data.

---

## Setting Up the Environment:

### Steps to Set Up the Project:

1. **Clone the Repository**: Clone the project repository to your local machine:
    ```bash
    git clone https://github.com/jgl0221/cs506-giddygavin.git
    cd cs506-giddygavin
    ```

2. **Create a Virtual Environment (Optional but Recommended)**: Isolate project dependencies in a virtual environment:
    ```bash
    python3 -m venv venv
    ```

    - On macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

    - On Windows:
    ```bash
    venv\Scripts\activate
    ```

3. **Install Dependencies**: Install the necessary dependencies via the Makefile or requirements.txt:

    - Using **Makefile**:
    ```bash
    make install
    ```

    - Or using **pip**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare the Data**: Ensure that the necessary data files (e.g., `Combined_data_2018_2022.csv`) are placed in the project directory. If they are not provided, download them according to the instructions.

---

## Running the Project:
Once the environment is set up, execute the project by running:
```bash
python app.py
```

This will run the main script, which loads the data, performs model predictions, and generates visualizations.

---

## Data Acquisition, Processing, and Cleaning

### Data Description and Relevance

The data used in this project comes from the World Happiness Report (2018–2022), a widely recognized source of global happiness metrics. The dataset includes multiple factors related to happiness and quality of life, such as GDP per capita, social support, life expectancy, freedom, and corruption levels. These factors are crucial in understanding the socio-economic determinants of happiness across different countries.

### Data Acquisition

The data was acquired from the official World Happiness Report website. We used the 2018–2022 reports for training our model and the 2023 data for testing.

### Data Preprocessing and Cleaning

- **Feature Selection:** We selected variables based on their relevance to happiness prediction. Features like GDP per capita, social support, and life expectancy were retained, while rank-based or irrelevant features were discarded.
- **Handling Missing Data:** We dropped countries not represented across all years (2018–2023) to ensure consistency in the dataset. This avoids introducing bias from missing data and ensures robust training and testing.
- **Normalization:** We normalized certain features (e.g., GDP per capita, life expectancy) to scale them between 0 and 1, preventing any feature from disproportionately affecting the model due to its scale.
- **Data Integrity:** After preprocessing, we ensured that the dataset contained no duplicates and that the target variable (`life_ladder`) had valid values for all entries.

---

## Exploratory Data Analysis (EDA) and Visualizations

### 1. Distribution of Key Features

We examined the distribution of key features using histograms and density plots:

- **Life Ladder (Happiness Score):** Symmetric distribution centered around the median happiness levels.
- **Normalized GDP per Capita & Healthy Life Expectancy:** Normally distributed after normalization, reflecting variation across countries.
- **Social Support, Freedom to Make Life Choices, Generosity:** Right-skewed distributions, suggesting lower values in many countries.
- **Corruption:** Strongly skewed with most countries reporting low corruption.

### 2. Relationships Between Features and Life Ladder

We used scatter plots to visualize the relationships between each explanatory variable and the Life Ladder:

- **Normalized GDP per Capita:** Strong positive correlation with happiness.
- **Social Support:** Positive correlation, reinforcing the importance of social connections.
- **Freedom to Make Life Choices:** Positive correlation, emphasizing that personal freedoms contribute to happiness.
- **Generosity & Corruption:** Weak positive relationship for generosity, and a negative relationship for corruption, showing that lower corruption levels are associated with higher happiness.
- **Normalized Healthy Life Expectancy:** Positive correlation, indicating that better health leads to higher happiness.

### 3. Correlation Heatmap

A heatmap was generated to quantify the correlation between features. Key observations:

- Strong positive correlations between `life_ladder` and `social_support`, `normalized_gdp_per_capita`, and `freedom_to_make_life_choices`.
- Low correlations between generosity and corruption, suggesting minimal multicollinearity.

---

## Methodologies

### 1. Model Selection and Training

We utilized the Random Forest Regressor as our primary model for predicting happiness scores. The model was trained using data from 2018–2022 and evaluated on the 2023 data. We assessed the model’s performance using the following metrics:

- **Mean Absolute Error (MAE):** 0.4758
- **Mean Squared Error (MSE):** 0.4161
- **Root Mean Squared Error (RMSE):** 0.6451
- **R-squared (R²):** 0.6724
- **Cross-validated MSE scores:** 0.2262 ± 0.0640

### 2. Feature Importance Analysis

We analyzed the importance of each feature using the Random Forest model:

- **Normalized Healthy Life Expectancy:** 0.6324
- **Normalized GDP per Capita:** 0.1182
- **Social Support:** 0.0923
- **Freedom to Make Life Choices:** 0.0638
- **Corruption:** 0.0565
- **Generosity:** 0.0367

---

## Model Comparison

We compared several machine learning models to predict happiness scores:

- **Gradient Boosting Regressor:** R² = 0.7299
- **XGBoost Regressor:** R² = 0.7226
- **LightGBM Regressor:** R² = 0.7385
- **Support Vector Regression (SVR):** R² = 0.7990

### Model Comparison Insights

- SVR was the most accurate model, with the highest R² and lowest RMSE.
- LightGBM performed well, followed by Gradient Boosting and XGBoost.
- SVR's excellent performance shows its ability to capture complex patterns in the data.

---

## Optimization and Evaluation

### 1. Hyperparameter Tuning for SVR

After identifying SVR as the best model, we performed GridSearchCV to optimize the hyperparameters for better accuracy.

### 2. Dimensionality Reduction with PCA

We applied Principal Component Analysis (PCA) to reduce feature dimensionality while retaining 95% of the variance in the data. The combination of PCA and SVR resulted in significant performance improvements.

### 3. Cross-Validation

We conducted 10-fold cross-validation to assess the model's generalizability. The mean R² score was 0.3921, with a standard deviation of 0.1669, and the validation set achieved an R² score of 0.8033.

---

## Results and Conclusion

### Key Findings:

- The SVR model outperformed other models, achieving the highest R² score (0.7990) and the lowest RMSE.
- PCA + SVR reduced dimensionality effectively, leading to better efficiency and performance with an R² score of 0.8033.
- Feature importance analysis highlighted normalized healthy life expectancy as the most influential factor in predicting happiness.

### Limitations:

- Dropping missing values may have resulted in a loss of important data, potentially introducing bias.
- PCA reduced feature interpretability, making it harder to understand the direct contribution of each variable.
- Cross-validation variability suggested the model may be sensitive to data subsets, indicating potential overfitting.

### Future Work:

- Incorporate external factors (e.g., political and cultural variables) for a more holistic understanding of happiness.
- Use advanced imputation techniques to handle missing data more effectively.
- Experiment with more advanced feature engineering and model ensembling methods to improve predictive performance.

---

## Interactive Visualizations

We provided several interactive visualizations to support the analysis:

- **Ladder Score Bar Chart:** Displays happiness scores for each country in a specific year.
- **Happiness Map:** A choropleth map showing global happiness scores, with color-coding for easy comparison.
- **Feature Statistics:** Bar charts for average and maximum values of key features such as GDP and social support.
- **Happiness Change Visualization:** Compares happiness scores between two years, highlighting significant changes.
- **Predicting Happiness Score for 2023:** Users can input values via sliders, predict happiness scores for 2023, and compare predictions with actual data.
