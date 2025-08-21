import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def simple_average_forecast(data, periods_ahead=1):
    """Calculates the simple average forecast."""
    avg = np.mean(data)
    return [avg] * periods_ahead

def moving_average_forecast(data, periods_ahead=1, window=4):
    """Calculates the moving average forecast."""
    if len(data) < window:
        return simple_average_forecast(data, periods_ahead)
    last_avg = np.mean(data[-window:])
    return [last_avg] * periods_ahead

def centered_moving_average_forecast(data, periods_ahead=1, window=4):
    """
    Calculates a forecast using the last valid value of a centered moving average.
    Note: This is an adaptation for forecasting, as CMA is typically a smoothing technique.
    """
    if len(data) < window:
        return simple_average_forecast(data, periods_ahead)
    
    cma = pd.Series(data).rolling(window=window, center=True).mean()
    last_valid_cma = cma.dropna().iloc[-1]
    return [last_valid_cma] * periods_ahead

def exponential_smoothing_forecast(data, periods_ahead=1, alpha=0.2):
    """Calculates the exponential smoothing forecast."""
    forecast = [data[0]]
    for i in range(1, len(data)):
        forecast.append(alpha * data[i] + (1 - alpha) * forecast[i-1])
    
    future_forecasts = []
    last_forecast = forecast[-1]
    for _ in range(periods_ahead):
        future_forecasts.append(last_forecast)
    return future_forecasts

def holts_method_forecast(data, periods_ahead=1, alpha=0.2, beta=0.2):
    """Calculates Holt's method forecast."""
    if len(data) < 2:
        return simple_average_forecast(data, periods_ahead)
    level = [data[0]]
    trend = [data[1] - data[0]]
    forecast = [level[0] + trend[0]]

    for i in range(1, len(data)):
        level.append(alpha * data[i] + (1 - alpha) * (level[i-1] + trend[i-1]))
        trend.append(beta * (level[i] - level[i-1]) + (1 - beta) * trend[i-1])
        forecast.append(level[i] + trend[i])

    future_forecasts = []
    last_level = level[-1]
    last_trend = trend[-1]
    for i in range(1, periods_ahead + 1):
        future_forecasts.append(last_level + i * last_trend)
    return future_forecasts

def winters_method_forecast(data, periods_ahead=1, season_length=4, alpha=0.2, beta=0.2, gamma=0.2):
    """Calculates Winter's method forecast."""
    if len(data) < season_length:
        return holts_method_forecast(data, periods_ahead, alpha, beta)
        
    level = [data[0]]
    trend = [0]
    season = list(data[:season_length])
    forecast = []

    for i in range(len(data)):
        if i >= season_length:
            level.append(alpha * (data[i] - season[i - season_length]) + (1 - alpha) * (level[i-1] + trend[i-1]))
            trend.append(beta * (level[i] - level[i-1]) + (1 - beta) * trend[i-1])
            season.append(gamma * (data[i] - level[i]) + (1 - gamma) * season[i - season_length])
        
        if i < season_length:
             forecast.append(level[-1] + trend[-1] + season[i])
        else:
             forecast.append(level[-1] + trend[-1] + season[i - season_length])


    future_forecasts = []
    last_level = level[-1]
    last_trend = trend[-1]
    for i in range(1, periods_ahead + 1):
        m = (i - 1) % season_length
        future_forecasts.append(last_level + i * last_trend + season[len(data) - season_length + m])
    return future_forecasts

def deseasonalize_forecast(data, periods_ahead=1, season_length=4):
    """Calculates a forecast using deseasonalization."""
    if len(data) < season_length:
        return linear_regression_forecast(data, periods_ahead)

    seasonal_indices = {}
    for i in range(season_length):
        seasonal_indices[i] = np.mean(data[i::season_length]) / np.mean(data)

    deseasonalized_data = [data[i] / seasonal_indices[i % season_length] for i in range(len(data))]

    X = np.arange(len(deseasonalized_data)).reshape(-1, 1)
    y = deseasonalized_data
    model = LinearRegression()
    model.fit(X, y)

    future_periods = np.arange(len(data), len(data) + periods_ahead).reshape(-1, 1)
    trend_forecast = model.predict(future_periods)
    
    seasonal_forecast = [trend_forecast[i] * seasonal_indices[(len(data) + i) % season_length] for i in range(periods_ahead)]
    return seasonal_forecast

def linear_regression_forecast(data, periods_ahead=1):
    """Calculates a forecast using linear regression."""
    X = np.arange(len(data)).reshape(-1, 1)
    y = data
    model = LinearRegression()
    model.fit(X, y)
    
    future_periods = np.arange(len(data), len(data) + periods_ahead).reshape(-1, 1)
    return model.predict(future_periods)

# --- Streamlit App ---

st.title("Time Series Forecasting")

num_data_points = st.number_input("Number of data points:", min_value=2, value=10)

demand_values = []
for i in range(num_data_points):
    demand_values.append(st.number_input(f"Demand value for period {i+1}:", value=float(100 + 20 * np.sin(i * np.pi/2) + i*2)))

method = st.selectbox("Select a forecasting method:", 
                      ("Simple Average", "Moving Average", "Centered Moving Average", "Exponential Smoothing", 
                       "Holt's Method", "Winter's Method", "Deseasonalization", 
                       "Linear Regression"))

forecast_values = st.number_input("How many forecast values are required?", min_value=1, value=4)

constants = {}
if method == "Moving Average" or method == "Centered Moving Average":
    constants['window'] = st.slider("Window:", 1, num_data_points, 4)
elif method == "Exponential Smoothing":
    constants['alpha'] = st.slider("Alpha:", 0.0, 1.0, 0.2)
elif method == "Holt's Method":
    constants['alpha'] = st.slider("Alpha:", 0.0, 1.0, 0.2)
    constants['beta'] = st.slider("Beta:", 0.0, 1.0, 0.2)
elif method == "Winter's Method":
    constants['alpha'] = st.slider("Alpha:", 0.0, 1.0, 0.2)
    constants['beta'] = st.slider("Beta:", 0.0, 1.0, 0.2)
    constants['gamma'] = st.slider("Gamma:", 0.0, 1.0, 0.2)
    constants['season_length'] = st.slider("Season Length:", 1, num_data_points, 4)
elif method == "Deseasonalization":
    constants['season_length'] = st.slider("Season Length:", 1, num_data_points, 4)

if st.button("Forecast"):
    forecast = []
    if method == "Simple Average":
        forecast = simple_average_forecast(demand_values, forecast_values)
    elif method == "Moving Average":
        forecast = moving_average_forecast(demand_values, forecast_values, **constants)
    elif method == "Centered Moving Average":
        forecast = centered_moving_average_forecast(demand_values, forecast_values, **constants)
    elif method == "Exponential Smoothing":
        forecast = exponential_smoothing_forecast(demand_values, forecast_values, **constants)
    elif method == "Holt's Method":
        forecast = holts_method_forecast(demand_values, forecast_values, **constants)
    elif method == "Winter's Method":
        forecast = winters_method_forecast(demand_values, forecast_values, **constants)
    elif method == "Deseasonalization":
        forecast = deseasonalize_forecast(demand_values, forecast_values, **constants)
    elif method == "Linear Regression":
        forecast = linear_regression_forecast(demand_values, forecast_values)
        
    st.write("Forecasted Values:")
    st.table(pd.DataFrame({"Period": range(num_data_points + 1, num_data_points + 1 + forecast_values),
                           "Forecast": forecast}))

    all_periods = list(range(1, num_data_points + 1))
    future_periods = list(range(num_data_points + 1, num_data_points + 1 + forecast_values))
    
    df = pd.DataFrame({
        'Period': all_periods + future_periods,
        'Demand': demand_values + [None] * forecast_values,
        'Forecast': [None] * num_data_points + forecast
    })

    st.line_chart(df.set_index('Period'))

