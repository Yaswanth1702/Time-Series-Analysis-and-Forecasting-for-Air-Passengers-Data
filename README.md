# Time Series Analysis of Air Passengers Data

This project is comprehensive analysis of the Air Passengers time series dataset using Python. The analysis covers data exploration, stationarity checks, transformations, and ARIMA modeling to forecast the number of airline passengers.

## Overview

The project performs the following steps:

1. **Data Loading**: Reads and preprocesses the Air Passengers dataset.
2. **Exploratory Data Analysis**: Plots and examines the time series data.
3. **Stationarity Check**: Utilizes visual methods and statistical tests to check if the data is stationary.
4. **Transformation**: Applies log transformation to stabilize variance.
5. **Trend and Seasonality Adjustment**:
   - Uses moving averages and exponentially weighted moving averages to handle trends.
   - Differencing and decomposition techniques are applied to handle seasonality.
6. **ARIMA Modeling**: Identifies the best ARIMA model using Auto ARIMA and fits the model.
7. **Model Evaluation**: Evaluates the model performance using RMSE.

## Usage

1. **Data Loading**:
   - Load the data using `pd.read_csv` with `parse_dates` and `index_col` parameters.

2. **Check Stationarity**:
   - Use `test_stationarity` function to visualize rolling statistics and perform the Dickey-Fuller test.

3. **Transformations**:
   - Apply log transformation and visualize using `plt.plot`.

4. **Adjust Trends and Seasonality**:
   - Use moving averages, exponential moving averages, differencing, and decomposition techniques.

5. **Model Building**:
   - Utilize `auto_arima` from the `pmdarima` library to determine the best ARIMA model.
   - Fit the ARIMA model and evaluate its performance.

## Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('AirPassengers.csv', parse_dates=True, index_col='Month')
ts = df['#Passengers']

# Check stationarity
test_stationarity(ts)

# Transform data
ts_log = np.log(ts)
moving_avg = ts_log.rolling(12).mean()

# Difference and decompose
ts_log_diff = ts_log - ts_log.shift()
decomposing = seasonal_decompose(ts_log)

# ARIMA Modeling
model = auto_arima(ts_log_diff.dropna(), start_p=1, start_q=1, seasonal=False, trace=True)
results = sm.tsa.arima.ARIMA(ts_log_diff.dropna(), order=(0, 1, 5)).fit()

# Evaluate model
error = np.sqrt(mean_squared_error(ts_log_diff.dropna(), results.fittedvalues))
print(f'Root Mean Squared Error: {error}')
```

## Results

- **Stationarity Test**: Provides results of Dickey-Fuller Test for stationarity.
- **Log Transformation**: Stabilized variance in the series.
- **Model Performance**: ARIMA model with parameters (0, 1, 5) was found optimal with an RMSE of 0.0928.
