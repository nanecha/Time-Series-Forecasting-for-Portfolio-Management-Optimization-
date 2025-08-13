import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import os

# ------------------------------
# 1. Load Data
# ------------------------------


def load_tesla_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    return df

# ------------------------------
# 2. Train-Test Split
# ------------------------------


def train_test_split(df, test_months=6):
    train_end = len(df) - test_months * 21  # Approx. 21 trading days per month
    train_data = df['Close'].iloc[:train_end]
    test_data = df['Close'].iloc[train_end:]
    print(f"Train data length: {len(train_data)}, Test data length: {len(test_data)}")
    return train_data, test_data

# ------------------------------
# 3. Fit ARIMA Model
# ------------------------------

def fit_arima_model(train_data):
    model = auto_arima(
        train_data,
        start_p=0, start_q=0,
        max_p=5, max_d=2, max_q=5,
        seasonal=True, m=12,
        stepwise=True, trace=True,
        error_action='ignore', suppress_warnings=True
    )
    return model

# ------------------------------
# 4. Forecast Future (6 months)
# ------------------------------

def forecast_future(model, last_date, months_ahead=6):
    periods = months_ahead * 21  # Approx. trading days in given months
    forecast, conf_int = model.predict(n_periods=periods, return_conf_int=True)

    forecast_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    forecast_series = pd.Series(forecast, index=forecast_index)
    conf_df = pd.DataFrame(conf_int, index=forecast_index, columns=['Lower', 'Upper'])
    return forecast_series, conf_df

# ------------------------------
# 5. Plot Forecast
# ------------------------------

def plot_forecast(train_data, test_data, forecast_series, conf_df, output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label='Training Data')
    plt.plot(test_data, label='Test Data', color='orange')
    plt.plot(forecast_series, label='6-Month Forecast', color='green')
    plt.fill_between(conf_df.index, conf_df['Lower'], conf_df['Upper'], color='pink', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Tesla Stock Price (USD)')
    plt.title('Tesla 6-Month Stock Price Forecast')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# ------------------------------
# 6. Analyze Forecast
# ------------------------------

def analyze_forecast(forecast_series, conf_df, last_actual_price):
    analysis = []
    analysis.append("📈 Tesla 6-Month Forecast Analysis")
    analysis.append("-" * 40)
    analysis.append(f"Last Actual Price: ${last_actual_price:.2f}")
    analysis.append(f"Predicted Price Range: ${forecast_series.min():.2f} - ${forecast_series.max():.2f}")
    analysis.append(f"Lowest Forecast (Date: {forecast_series.idxmin().date()}): ${forecast_series.min():.2f}")
    analysis.append(f"Highest Forecast (Date: {forecast_series.idxmax().date()}): ${forecast_series.max():.2f}")
    
    pct_change = ((forecast_series.iloc[-1] - last_actual_price) / last_actual_price) * 100
    analysis.append(f"Expected % Change in 6 Months: {pct_change:.2f}%")

    conf_low = conf_df['Lower'].min()
    conf_high = conf_df['Upper'].max()
    analysis.append(f"Confidence Interval Range: ${conf_low:.2f} - ${conf_high:.2f}")
    return "\n".join(analysis)

# ------------------------------
# 7. Main Execution
# ------------------------------


if __name__ == "__main__":
    pass
