```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import os
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(data_path, end_date='2025-07-30'):
    """
    Load and prepare TSLA data for forecasting.
    
    Parameters:
    -----------
    data_path : str
        Path to TSLA_data.csv.
    end_date : str
        End date for data (YYYY-MM-DD).
    
    Returns:
    --------
    data : Series
        Adjusted Close prices.
    """
    try:
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        adj_close = df['Adj Close']
        
        # Handle missing values
        adj_close = adj_close.fillna(method='ffill').fillna(method='bfill')
        data = adj_close[:end_date]
        
        if data.empty:
            raise ValueError("Data is empty. Check date range or data availability.")
        
        print(f"Data shape: {data.shape}")
        print(f"Data date range: {data.index.min()} to {data.index.max()}")
        print(f"NaN check: {data.isna().sum()}")
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def fit_arima_model(data):
    """
    Fit SARIMAX(0,1,0)(2,0,0)[12] model on the data.
    
    Parameters:
    -----------
    data : Series
        Adjusted Close prices.
    
    Returns:
    --------
    arima_model : SARIMAXResults
        Fitted ARIMA model.
    """
    try:
        # Fit SARIMAX(0,1,0)(2,0,0)[12]
        arima_model = SARIMAX(data, order=(0,1,0), seasonal_order=(2,0,0,12))
        arima_fit = arima_model.fit(disp=False)
        
        print("ARIMA model fitted successfully.")
        print(arima_fit.summary())
        
        return arima_fit
    except Exception as e:
        print(f"Error fitting ARIMA: {e}")
        return None

def forecast_arima(model, forecast_steps=252, start_date='2025-07-31'):
    """
    Generate ARIMA forecast with confidence intervals.
    
    Parameters:
    -----------
    model : SARIMAXResults
        Fitted ARIMA model.
    forecast_steps : int
        Number of steps to forecast (252 trading days ~ 12 months).
    start_date : str
        Start date for forecast (YYYY-MM-DD).
    
    Returns:
    --------
    forecast_df : DataFrame
        Forecasted values and confidence intervals.
    """
    try:
        # Generate forecast and confidence intervals
        forecast_result = model.get_forecast(steps=forecast_steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence intervals
        
        # Create datetime index for forecast
        start_date = pd.to_datetime(start_date)
        forecast_dates = pd.date_range(start=start_date, periods=forecast_steps, freq='B')  # Business days
        
        # Create DataFrame
        forecast_df = pd.DataFrame({
            'Forecast': forecast_mean,
            'Lower_CI': conf_int.iloc[:, 0],
            'Upper_CI': conf_int.iloc[:, 1]
        }, index=forecast_dates)
        
        print(f"Forecast length: {len(forecast_df)}")
        print(f"Forecast date range: {forecast_df.index.min()} to {forecast_df.index.max()}")
        print(f"NaN check: {forecast_df.isna().sum()}")
        
        return forecast_df
    except Exception as e:
        print(f"Error generating forecast: {e}")
        return None

def visualize_forecast(historical_data, forecast_df, output_dir='plots/forecasting'):
    """
    Visualize historical data, forecast, and confidence intervals.
    
    Parameters:
    -----------
    historical_data : Series
        Historical Adjusted Close prices.
    forecast_df : DataFrame
        Forecasted values and confidence intervals.
    output_dir : str
        Directory to save plot.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        # Plot historical data
        plt.plot(historical_data.index, historical_data, label='Historical Data', color='blue')
        # Plot forecast
        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='green')
        # Plot confidence intervals
        plt.fill_between(forecast_df.index, forecast_df['Lower_CI'], forecast_df['Upper_CI'],
                         color='green', alpha=0.2, label='95% Confidence Interval')
        
        plt.title('TSLA Stock Price Forecast (August 2025 - August 2026)')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price (USD)')
        plt.legend()
        plt.grid()
        plt.savefig(f'{output_dir}/task3_forecast.png')
        plt.close()
        
        print(f"Forecast plot saved to {output_dir}/task3_forecast.png")
    except Exception as e:
        print(f"Error visualizing forecast: {e}")

def analyze_forecast(forecast_df, output_dir='data/output'):
    """
    Analyze forecast trends, volatility, and market implications.
    
    Parameters:
    -----------
    forecast_df : DataFrame
        Forecasted values and confidence intervals.
    output_dir : str
        Directory to save analysis.
    
    Returns:
    --------
    analysis : str
        Text summary of trend, volatility, and market insights.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Trend Analysis
        forecast_mean = forecast_df['Forecast']
        trend_slope = (forecast_mean[-1] - forecast_mean[0]) / len(forecast_mean)
        trend = "upward" if trend_slope > 0 else "downward" if trend_slope < 0 else "stable"
        trend_description = f"The forecast exhibits a {trend} trend with a slope of {trend_slope:.2f} USD per trading day."
        
        # Check for anomalies (e.g., extreme changes)
        forecast_diff = forecast_mean.diff().abs()
        anomaly_threshold = forecast_diff.mean() + 2 * forecast_diff.std()
        anomalies = forecast_diff[forecast_diff > anomaly_threshold]
        anomaly_description = f"Anomalies detected: {len(anomalies)} significant daily changes exceeding {anomaly_threshold:.2f} USD."
        
        # Volatility and Risk (Confidence Intervals)
        ci_width = forecast_df['Upper_CI'] - forecast_df['Lower_CI']
        ci_width_mean = ci_width.mean()
        ci_width_trend = (ci_width[-1] - ci_width[0]) / len(ci_width)
        volatility_description = (f"The average confidence interval width is {ci_width_mean:.2f} USD. "
                                f"The width {'increases' if ci_width_trend > 0 else 'decreases'} over time "
                                f"by {ci_width_trend:.2f} USD per trading day, indicating "
                                f"{'increasing' if ci_width_trend > 0 else 'decreasing'} uncertainty in long-term forecasts.")
        
        # Market Opportunities and Risks
        price_change = (forecast_mean[-1] - forecast_mean[0]) / forecast_mean[0] * 100
        opportunities_risks = (f"Market Opportunities: A {trend} trend suggests "
                              f"{'potential buying opportunities if prices rise (forecasted change: {price_change:.2f}%)' if price_change > 0 else 'caution due to potential declines (forecasted change: {price_change:.2f}%)'}.")
        risks = (f"Market Risks: High volatility (CI width: {ci_width_mean:.2f} USD) and "
                 f"{'increasing' if ci_width_trend > 0 else 'decreasing'} uncertainty pose risks for long-term investments. "
                 f"Anomalies ({len(anomalies)}) may indicate sudden price movements.")
        
        # Combine analysis
        analysis = "\n".join([
            "Task 3: Forecast Analysis for TSLA (August 2025 - August 2026)",
            "=" * 60,
            "Trend Analysis:",
            trend_description,
            anomaly_description,
            "\nVolatility and Risk:",
            volatility_description,
            "\nMarket Opportunities and Risks:",
            opportunities_risks,
            risks
        ])
        
        # Save analysis
        with open(f'{output_dir}/task3_analysis.txt', 'w') as f:
            f.write(analysis)
        
        print("Analysis saved to data/output/task3_analysis.txt")
        print("\n" + analysis)
        
        # Save forecast data
        forecast_df.to_csv(f'{output_dir}/task3_forecast.csv')
        print(f"Forecast data saved to {output_dir}/task3_forecast.csv")
        
        return analysis
    except Exception as e:
        print(f"Error analyzing forecast: {e}")
        return None

def main():
    """
    Main function to execute Task 3 forecasting and analysis.
    """
    # Parameters
    data_path = 'F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/TSLA_data.csv'
    end_date = '2025-07-30'
    forecast_steps = 252  # 12 months of trading days
    output_dir = 'data/output'
    plot_dir = 'plots/forecasting'
    
    # Load and prepare data
    data = load_and_prepare_data(data_path, end_date)
    if data is None:
        return
    
    # Fit ARIMA model
    arima_model = fit_arima_model(data)
    if arima_model is None:
        return
    
    # Generate forecast
    forecast_df = forecast_arima(arima_model, forecast_steps=forecast_steps, start_date='2025-07-31')
    if forecast_df is None:
        return
    
    # Visualize forecast
    visualize_forecast(data, forecast_df, output_dir=plot_dir)
    
    # Analyze forecast
    analyze_forecast(forecast_df, output_dir=output_dir)

if __name__ == "__main__":
    pass
