# Project Overview: Predicting Happiness Scores with World Happiness Report (November 5th, 2024)

**Presentation Link**: [Watch the Video](https://youtu.be/5C-bmi80cLQ)

This project aims to analyze World Happiness Report data from 2020, 2021, and 2022 to model and predict happiness scores (referred to as "life ladder" values). Using socio-economic indicators like GDP per capita, social support, life expectancy, freedom of life choices, generosity, and perceptions of corruption, we seek to understand the relationships between these factors and national happiness levels. Through a Random Forest Regressor model, we also identify the most influential predictors contributing to happiness.

## Table of Contents
1. [Data Processing and Standardization](#data-processing-and-standardization)
2. [Preliminary Data Visualizations](#preliminary-data-visualizations)
3. [Modeling Approach: Random Forest Regressor](#modeling-approach-random-forest-regressor)
4. [Feature Importance Analysis](#feature-importance-analysis)
5. [Next Steps](#next-steps)

---

### 1. Data Processing and Standardization
The data processing phase included:

- **Standardizing Column Names**: We imported datasets from each year and standardized column names for consistency.
- **Aligning Features Across Years**: After adding a "year" column to capture time-specific details, we aligned the datasets by renaming columns and selecting common features.
- **Handling Missing Values**: We checked for missing data and removed rows with missing entries to ensure a cleaner dataset.

These steps helped us create a consolidated dataset ready for analysis and modeling.

---

### 2. Preliminary Data Visualizations
To explore the data, we created several visualizations to understand feature distributions and relationships with happiness:

- **Distribution Plots**: Histograms and density plots provided insights into the spread of key features such as GDP per capita, social support, and freedom, highlighting variability across countries and years.
- **Scatter Plots**: We examined the relationships between happiness (life ladder) and other features, finding positive correlations with GDP per capita, life expectancy, and social support.
- **Correlation Heatmap**: A correlation heatmap revealed stronger correlations between happiness and features like GDP per capita and social support, suggesting these as influential variables.

These visualizations provided a foundation for understanding which factors might drive happiness.

---

### 3. Modeling Approach: Random Forest Regressor
For our initial model, we chose the **Random Forest Regressor**, ideal for handling non-linear relationships and providing feature importance rankings.

- **Data Preparation**: We split the data into training and testing sets, applying Min-Max scaling to normalize feature values for model consistency.
- **Model Training**: We trained the model using 80% of the data and evaluated it on the remaining 20%.
- **Evaluation Metrics**:
  - **Mean Absolute Error (MAE)**: Measures the average prediction error, indicating how close predictions are to actual values.
  - **R-squared (R²)**: Reflects the proportion of variance in happiness scores explained by the model.

The model’s performance is promising, achieving decent accuracy based on initial evaluation metrics.

---

### 4. Feature Importance Analysis
Using Random Forest’s feature importance metric, we identified key predictors of happiness:

- **Top Predictors**: The strongest predictors were `Log GDP per capita`, `social support`, and `healthy life expectancy`.
- **Less Influential Features**: Generosity and perceptions of corruption had weaker importance scores, suggesting they may influence happiness less than economic and social factors.

This analysis provides valuable insights into the most impactful variables, helping guide future modeling steps.

---

### 5. Next Steps
To refine our model, we plan to:

- **Tune Model Parameters**: Adjust Random Forest parameters (e.g., number of trees, tree depth) to balance complexity and accuracy.
- **Mitigate Overfitting**: Use cross-validation and potentially limit tree depth if overfitting becomes a concern.
- **Explore Feature Interactions**: Investigate interactions between predictors, as combined effects of features like social support and freedom of choice could reveal deeper insights.
- **Expand Dataset**: Consider integrating additional socio-economic indicators if available.

---

This project is an ongoing effort to understand what drives happiness across nations, using data-driven insights to identify the strongest contributing factors. By continuing to refine our model, we aim to provide actionable recommendations for policymakers and further contribute to the study of well-being.


# Project Proposal: Predicting a Country's Happiness Index (October 1st, 2024)

## Description of the Project
The goal of this project is to predict a country's **happiness index** based on socio-economic and demographic factors derived from the World Happiness Report dataset available on Kaggle. The happiness index reflects the subjective well-being of individuals in various countries, influenced by factors such as GDP per capita, social support, life expectancy, freedom to make life choices, and perceptions of corruption. This analysis will provide insights into what drives happiness across different nations and help inform policy decisions aimed at improving the quality of life.

## Goals
- **Develop a Predictive Model:** Create a model that can accurately predict a country’s happiness index using key socio-economic indicators. This will help identify which factors are the strongest predictors of happiness. We will aim to predict the happiness index for the year 2023 using a model trained on previous years.
  
- **Identify Key Factors Influencing Happiness:** Analyze the dataset to determine the significant variables that impact happiness levels. This could involve exploring correlations and conducting feature importance analysis.

## Data Collection

### Primary Data Source
The main dataset will be sourced from Kaggle's **World Happiness Report**, which includes annual happiness scores for various countries along with associated factors like GDP per capita, social support, life expectancy, freedom to make life choices, and perceptions of corruption.

### Data Features
The primary socio-economic indicators used for predicting the happiness index include:

- **Economic indicators:** GDP per capita (gross domestic product per person).
- **Social factors:** Social support (strength of social connections and support networks).
- **Life expectancy:** The average number of years a person is expected to live.
- **Freedom to make life choices:** The extent to which people feel free to make their own decisions.
- **Perceptions of corruption:** How corrupt people perceive their government and businesses to be.

Additional features like **generosity** or **demographic indicators** (e.g., population growth rate) may be considered, depending on their relevance to happiness.

### Important Note
For each feature, we will align with how these indicators are typically measured in happiness studies:
- **GDP per capita:** Generally calculated by dividing a country’s total economic output by its population.
- **Social support:** Measured based on survey responses regarding the availability of support from friends or family in times of need.
- **Life expectancy:** Based on World Health Organization or similar health statistics.
- **Freedom to make life choices:** Gauged by survey questions that ask respondents whether they feel they have the freedom to make important life decisions.
- **Perceptions of corruption:** Derived from surveys asking about the level of corruption in government and businesses as perceived by citizens.

We will avoid using any factor that has already been included in the calculation of the happiness index itself to prevent redundancy.

### Data Collection Method
The Kaggle dataset will be downloaded directly. To ensure a comprehensive analysis, we may also seek additional relevant data from:
- **Web Scraping:** If needed, we will scrape data from reliable sources such as the World Bank and the United Nations.
- **APIs:** Explore APIs from organizations like the World Bank to gather updated information about various socio-economic indicators.

### Data Cleaning
The collected data will be preprocessed to:
- Handle missing values
- Remove duplicates
- Normalize data formats

This step is crucial to ensure the reliability of the predictive model.

## Modeling Approach

### Data Preparation
- Clean and preprocess the data, including handling missing values, normalizing data, and encoding categorical variables.

### Modeling Techniques
- We will begin with **linear regression** for baseline predictions.
- Explore advanced models such as **decision trees** or **random forests** to capture more complex relationships, as well as other methods learned in class.
- Consider using **neural networks** if data complexity warrants such an approach.

### Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **R-squared**

## Data Visualization

### Visualization Techniques
- **Scatter plots** to illustrate relationships between individual factors and the happiness index.
- **Correlation heatmaps** to identify strong relationships among variables.

### Exploratory Data Analysis (EDA)
We will conduct EDA to uncover patterns and insights within the dataset through various visualizations.

## Test Plan

### Data Splitting
The dataset will be split into **training** and **testing** sets to evaluate model predictive capabilities.

## Conclusion
This project aims to leverage the insights from the **World Happiness Report** dataset to create a predictive model of a country's happiness index. By identifying key factors that influence happiness and generating actionable recommendations for policymakers, we hope to contribute to the ongoing discourse around improving well-being. The project will combine data analysis, modeling techniques, and effective visualizations to provide an understanding of what drives happiness across different nations.
