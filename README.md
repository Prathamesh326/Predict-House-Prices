
# Predict House Prices

This project uses the **Boston Housing Dataset** to predict house prices using multivariable linear regression. The model incorporates various data transformation techniques, including a **log transformation** of the target variable to improve model fit and accuracy.

## Table of Contents
- [Introduction](#introduction)
- [Data Overview](#data-overview)
- [Modeling Process](#modeling-process)
  - [Residual Analysis](#residual-analysis)
  - [Data Transformation](#data-transformation)
  - [Regression with Log-Transformed Prices](#regression-with-log-transformed-prices)
  - [Comparing Model Performance](#comparing-model-performance)
- [Predicting House Prices](#predicting-house-prices)
- [Conclusion](#conclusion)
- [Installation](#installation)

## Introduction

The goal of this project is to predict house prices using features like the number of rooms, crime rate, proximity to the river, pollution levels, and more. Initially, a linear regression model is fitted to the data. However, further analysis revealed that the target variable (`PRICE`) exhibited skewness, which motivated us to apply a log transformation to the target variable to achieve a better fit.

## Data Overview

The dataset consists of 13 features related to Boston homes, such as:
- `RM`: Number of rooms per dwelling
- `NOX`: Nitric Oxide concentration (pollution)
- `DIS`: Distance to employment centers
- `CHAS`: Charles River dummy variable (1 if next to the river)
- `PTRATIO`: Pupil-teacher ratio by town
- `LSTAT`: Percentage of the lower status population

The target variable is `PRICE`, which represents the value of homes in $1000s.

## Modeling Process

### Residual Analysis

We start by fitting a standard linear regression model and performing residual analysis to evaluate model performance. In particular, we check:
- **Residuals vs. Predicted values**: Ideally, residuals should show no pattern, indicating that the model does not have systematic biases.
- **Distribution of residuals**: A skewness and mean near zero indicate that the residuals follow a normal distribution, which is desirable for linear models.

### Data Transformation

After analyzing the residuals and the distribution of `PRICE`, we applied a log transformation to the target variable. The log transformation helps reduce the skewness of the target variable, leading to a more linear relationship between features and the target.

### Regression with Log-Transformed Prices

Using the transformed target variable, we refit the regression model. The new equation becomes:

$$ \log(\hat{PRICE}) = \theta_0 + \theta_1 RM + \theta_2 NOX + \theta_3 DIS + ... + \theta_{13} LSTAT $$

The log transformation compresses larger house prices, improving the model’s performance on the dataset. We then compared the new model to the original one using:
- **R-squared values**: On both training and test datasets.
- **Residual analysis**: Comparing skewness and mean of residuals for both models.

### Comparing Model Performance

We evaluated the models based on the **R-squared** values on the training and test sets. Additionally, we examined the residuals for each model to ensure that the log transformation resulted in a more normally distributed residual pattern.

- **Original Model R-squared (Training)**: `...`
- **Log-Transformed Model R-squared (Training)**: `...`
- **Original Model R-squared (Test)**: `...`
- **Log-Transformed Model R-squared (Test)**: `...`

## Predicting House Prices

Using the final log-transformed model, we predicted the price of an average property and other custom scenarios, such as properties near the river, with a specified number of rooms, etc.

### Example Predictions

#### Average Property:
```python
# Log Price Estimate
log_estimate = ...

# Convert to Dollar Estimate
dollar_estimate = np.exp(log_estimate) * 1000
```

#### Custom Property (next to river, 8 rooms, low poverty):
```python
# Define Property Characteristics
next_to_river = True
nr_rooms = 8
students_per_classroom = 20
distance_to_town = 5
pollution = data.NOX.quantile(q=0.75)
amount_of_poverty = data.LSTAT.quantile(q=0.25)

# Predict
log_estimate = ...
dollar_estimate = np.exp(log_estimate) * 1000
```

## Conclusion

Applying a log transformation significantly improved the model’s performance. Both the R-squared values and the residuals indicate a better fit for the log-transformed model. This project demonstrates the importance of data transformation and residual analysis in regression modeling.

## Installation

To run this project locally, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Prathamesh326/Predict-House-Prices.git
cd Predict-House-Prices
```

Then, you can open the notebook and follow the steps to replicate the analysis.

```bash
jupyter notebook
```

## License

This project is licensed under the MIT License.
