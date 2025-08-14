# src/time_series_forcasting_models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
from datetime import datetime


def prepare_data(data_path, train_end='2023-12-31', test_start='2024-01-01', max_test_end='2025-08-13'):
    """
    Load and split TSLA data into training and testing sets, limiting test period to available data.
    
    Parameters:
    -----------
    data_path : str
        Path to TSLA_data.csv.
    train_end : str
        End date for training data (YYYY-MM-DD).
    test_start : str
        Start date for testing data (YYYY-MM-DD).
    max_test_end : str
        Maximum end date for test data to avoid future dates (YYYY-MM-DD).
    
    Returns:
    --------
    train_data : Series
        Training Adjusted Close prices.
    test_data : Series
        Testing Adjusted Close prices.
    """
    try:
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        adj_close = df['Close']
        
        # Handle missing values
        adj_close = adj_close.fillna(method='ffill').fillna(method='bfill')
        
        train_data = adj_close[:train_end]
        test_data = adj_close[test_start:max_test_end]
        
        if train_data.empty or test_data.empty:
            raise ValueError("Training or testing data is empty. Check date range or data availability.")
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Testing data shape: {test_data.shape}")
        print(f"Test data date range: {test_data.index.min()} to {test_data.index.max()}")
        
        # Check for NaN
        if train_data.isna().any() or test_data.isna().any():
            raise ValueError("NaN values found in train or test data after cleaning.")
        
        return train_data, test_data
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None


def fit_arima(train_data, test_data, output_dir='data/output'):
    """
    Fit ARIMA model and forecast.

    Parameters:
    -----------
    train_data : Series
        Training data (Adjusted Close prices).
    test_data : Series
        Testing data (Adjusted Close prices) - needed for index alignment.
    output_dir : str
        Directory to save forecast CSV.

    Returns:
    --------
    arima_forecast : Series
        Forecasted values with test index.
    arima_model : ARIMA
        Fitted ARIMA model.
    """
    os.makedirs(output_dir, exist_ok=True)

    arima_model = auto_arima(train_data, start_p=0, start_q=0, max_p=5, max_d=2, max_q=5,
                             seasonal=True, m=12, stepwise=True, trace=True, error_action='warn',
                             suppress_warnings=True)

    arima_forecast_values = arima_model.predict(n_periods=len(test_data))

    print(f"Length of ARIMA forecast values: {len(arima_forecast_values)}")
    print(f"Length of test data index: {len(test_data.index)}")
    print(f"First few forecast values: {arima_forecast_values[:5]}")


    arima_forecast = pd.Series(arima_forecast_values) # Create Series from values first
    arima_forecast.index = test_data.index # Explicitly assign the index

    arima_forecast.to_csv(f'{output_dir}/arima_forecast.csv')
    return arima_forecast, arima_model


def evaluate_models(test_data, arima_forecast, lstm_forecast, output_dir='data/output'):
    """
    Evaluate ARIMA and LSTM models using MAE, RMSE, MAPE.

    Parameters:
    -----------
    test_data : Series
        Actual test data.
    arima_forecast : Series
        ARIMA predictions.
    lstm_forecast : Series
        LSTM predictions.
    output_dir : str
        Directory to save metrics CSV.

    Returns:
    --------
    metrics_df : DataFrame
        Evaluation metrics for both models.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics_data = {'Model': [], 'MAE': [], 'RMSE': [], 'MAPE': []}

    # Evaluate LSTM
    # Ensure indices are aligned for LSTM evaluation
    common_index_lstm = test_data.index.intersection(lstm_forecast.index)
    test_data_lstm_aligned = test_data[common_index_lstm]
    lstm_forecast_aligned = lstm_forecast[common_index_lstm]

    # Drop any potential NaNs that might still exist after alignment
    combined_lstm = pd.DataFrame({'actual': test_data_lstm_aligned, 'forecast': lstm_forecast_aligned}).dropna()
    test_data_lstm_aligned = combined_lstm['actual']
    lstm_forecast_aligned = combined_lstm['forecast']


    if not lstm_forecast_aligned.empty:
        lstm_mae = mean_absolute_error(test_data_lstm_aligned, lstm_forecast_aligned)
        lstm_rmse = np.sqrt(mean_squared_error(test_data_lstm_aligned, lstm_forecast_aligned))
        # Avoid division by zero in MAPE
        mape_dividend = np.abs((test_data_lstm_aligned - lstm_forecast_aligned))
        mape_divisor = np.abs(test_data_lstm_aligned)
        # Replace zero divisor with a small number to avoid inf/NaN
        mape_divisor[mape_divisor == 0] = 1e-10
        lstm_mape = np.mean(mape_dividend / mape_divisor) * 100

        metrics_data['Model'].append('LSTM')
        metrics_data['MAE'].append(lstm_mae)
        metrics_data['RMSE'].append(lstm_rmse)
        metrics_data['MAPE'].append(lstm_mape)
    else:
        print("LSTM forecast is empty after alignment and dropping NaNs, skipping LSTM evaluation.")


    # Evaluate ARIMA if forecast is not empty
    if not arima_forecast.empty:
        # Ensure indices are aligned for ARIMA evaluation
        common_index_arima = test_data.index.intersection(arima_forecast.index)
        test_data_arima_aligned = test_data[common_index_arima]
        arima_forecast_aligned = arima_forecast[common_index_arima]

        # Drop any potential NaNs that might still exist after alignment
        combined_arima = pd.DataFrame({'actual': test_data_arima_aligned, 'forecast': arima_forecast_aligned}).dropna()
        test_data_arima_aligned = combined_arima['actual']
        arima_forecast_aligned = combined_arima['forecast']

        if not arima_forecast_aligned.empty:
            arima_mae = mean_absolute_error(test_data_arima_aligned, arima_forecast_aligned)
            arima_rmse = np.sqrt(mean_squared_error(test_data_arima_aligned, arima_forecast_aligned))
             # Avoid division by zero in MAPE
            mape_dividend = np.abs((test_data_arima_aligned - arima_forecast_aligned))
            mape_divisor = np.abs(test_data_arima_aligned)
            # Replace zero divisor with a small number to avoid inf/NaN
            mape_divisor[mape_divisor == 0] = 1e-10
            arima_mape = np.mean(mape_dividend / mape_divisor) * 100

            metrics_data['Model'].append('ARIMA')
            metrics_data['MAE'].append(arima_mae)
            metrics_data['RMSE'].append(arima_rmse)
            metrics_data['MAPE'].append(arima_mape)
        else:
            print("ARIMA forecast is empty after alignment and dropping NaNs, skipping ARIMA evaluation.")


    metrics_df = pd.DataFrame(metrics_data)

    metrics_df.to_csv(f'{output_dir}/evaluation_metrics.csv', index=False)

    print("\nModel Performance Comparison:")
    display(metrics_df)

    return metrics_df

def plot_forecasts(train_data, test_data, arima_forecast, lstm_forecast, output_dir='plots/forecasting'):
    """
    Plot actual vs. forecasted prices for ARIMA and LSTM.

    Parameters:
    -----------
    train_data : Series
        Training data.
    test_data : Series
        Testing data.
    arima_forecast : Series
        ARIMA predictions.
    lstm_forecast : Series
        LSTM predictions.
    output_dir : str
        Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot ARIMA if forecast is not empty
    if not arima_forecast.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(train_data, label='Training Data', color='blue', alpha=0.7)
        plt.plot(test_data, label='Actual Test Data', color='green')
        plt.plot(arima_forecast, label='ARIMA Forecast', color='red', linestyle='--')
        plt.title('ARIMA Forecast vs Actual (TSLA)')
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/arima_forecast.png')
        plt.close()
    else:
        print("ARIMA forecast is empty, skipping ARIMA plot.")


    # Plot LSTM
    if not lstm_forecast.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(train_data, label='Training Data', color='blue', alpha=0.7)
        plt.plot(test_data, label='Actual Test Data', color='green')
        plt.plot(lstm_forecast, label='LSTM Forecast', color='purple', linestyle='--')
        plt.title('LSTM Forecast vs Actual (TSLA)')
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/lstm_forecast.png')
        plt.close()
    else:
        print("LSTM forecast is empty, skipping LSTM plot.")

    # Plot combined forecasts if both are not empty
    if not arima_forecast.empty and not lstm_forecast.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(test_data, label='Actual Test Data', color='green')
        plt.plot(arima_forecast, label='ARIMA Forecast', color='red', linestyle='--')
        plt.plot(lstm_forecast, label='LSTM Forecast', color='purple', linestyle='-.')
        plt.title('ARIMA and LSTM Forecasts vs Actual (TSLA)')
        plt.xlabel('Date')
        plt.ylabel('Close Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/combined_forecasts.png')
        plt.close()


    # Plot metrics if metrics_df is not empty
    try:
        metrics_df = pd.read_csv('data/output/evaluation_metrics.csv')
        if not metrics_df.empty:
            plt.figure(figsize=(10, 6))
            metrics_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
            plt.title('Model Performance Comparison (TSLA)')
            plt.ylabel('Metric Value')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/model_comparison.png')
            plt.close()
        else:
            print("Metrics DataFrame is empty, skipping metrics plot.")
    except FileNotFoundError:
        print("Evaluation metrics file not found, skipping metrics plot.")


def evaluate_models(test_data, arima_forecast, lstm_forecast, output_dir='data/output'):
    """
    Evaluate ARIMA and LSTM models using MAE, RMSE, MAPE.
    
    Parameters:
    -----------
    test_data : Series
        Actual test data.
    arima_forecast : Series
        ARIMA predictions.
    lstm_forecast : Series
        LSTM predictions.
    output_dir : str
        Directory to save metrics CSV.
    
    Returns:
    --------
    metrics_df : DataFrame
        Evaluation metrics for both models.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate inputs
        if test_data is None or arima_forecast is None or lstm_forecast is None:
            raise ValueError("One or more input series (test_data, arima_forecast, lstm_forecast) is None.")
        
        # Align indices
        common_index = test_data.index.intersection(arima_forecast.index).intersection(lstm_forecast.index)
        if len(common_index) == 0:
            raise ValueError("No common index found between test_data, arima_forecast, and lstm_forecast.")
        
        test_data = test_data[common_index]
        arima_forecast = arima_forecast[common_index]
        lstm_forecast = lstm_forecast[common_index]
        
        # Check for NaN values
        if test_data.isna().any():
            raise ValueError(f"NaN values found in test_data: {test_data.isna().sum()}")
        if arima_forecast.isna().any():
            raise ValueError(f"NaN values found in arima_forecast: {arima_forecast.isna().sum()}")
        if lstm_forecast.isna().any():
            raise ValueError(f"NaN values found in lstm_forecast: {lstm_forecast.isna().sum()}")
        
        # Calculate metrics with updated parameter
        arima_mae = mean_absolute_error(test_data, arima_forecast, ensure_all_finite=True)
        arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast, ensure_all_finite=True))
        arima_mape = np.mean(np.abs((test_data - arima_forecast) / test_data)) * 100
        
        lstm_mae = mean_absolute_error(test_data, lstm_forecast, ensure_all_finite=True)
        lstm_rmse = np.sqrt(mean_squared_error(test_data, lstm_forecast, ensure_all_finite=True))
        lstm_mape = np.mean(np.abs((test_data - lstm_forecast) / test_data)) * 100
        
        metrics_df = pd.DataFrame({
            'Model': ['ARIMA', 'LSTM'],
            'MAE': [arima_mae, lstm_mae],
            'RMSE': [arima_rmse, lstm_rmse],
            'MAPE': [arima_mape, lstm_mape]
        })
        
        metrics_df.to_csv(f'{output_dir}/evaluation_metrics.csv', index=False)
        
        print("\nModel Performance Comparison:")
        print(metrics_df)
        
        return metrics_df
    except Exception as e:
        print(f"Error evaluating models: {e}")
        return None


def plot_forecasts(train_data, test_data, arima_forecast, lstm_forecast, output_dir='plots/forecasting'):
    """
    Plot actual vs. forecasted prices for ARIMA and LSTM.
    
    Parameters:
    -----------
    train_data : Series
        Training data.
    test_data : Series
        Testing data.
    arima_forecast : Series
        ARIMA predictions.
    lstm_forecast : Series
        LSTM predictions.
    output_dir : str
        Directory to save plots.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate inputs
        if train_data is None or test_data is None or arima_forecast is None or lstm_forecast is None:
            raise ValueError("One or more input series (train_data, test_data, arima_forecast, lstm_forecast) is None.")
        
        # Plot ARIMA
        plt.figure(figsize=(12, 6))
        plt.plot(train_data, label='Training Data')
        plt.plot(test_data, label='Actual Test Data')
        plt.plot(arima_forecast, label='ARIMA Forecast')
        plt.title('ARIMA Forecast vs Actual (TSLA)')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price (USD)')
        plt.legend()
        plt.grid()
        plt.savefig(f'{output_dir}/arima_forecast.png')
        plt.show()
        
        # Plot LSTM
        plt.figure(figsize=(12, 6))
        plt.plot(train_data, label='Training Data')
        plt.plot(test_data, label='Actual Test Data')
        plt.plot(lstm_forecast, label='LSTM Forecast')
        plt.title('LSTM Forecast vs Actual (TSLA)')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price (USD)')
        plt.legend()
        plt.grid()
        plt.savefig(f'{output_dir}/lstm_forecast.png')
        plt.show()
        
        # Plot metrics
        metrics_df = pd.read_csv(f'{output_dir.replace("plots/forecasting", "data/output")}/evaluation_metrics.csv')
        plt.figure(figsize=(10, 6))
        metrics_df.set_index('Model').plot(kind='bar')
        plt.title('Model Performance Comparison (TSLA)')
        plt.ylabel('Metric Value')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png')
        plt.show()
    except Exception as e:
        print(f"Error plotting forecasts: {e}")

        
if __name__ == "__main__":
    pass
    # Example usage
    # data_path = 'F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/TSLA_data.csv'
    # train_data, test_data = prepare_data(data_path)
    # arima_forecast, arima_model = fit_arima(train_data)
    # lstm_forecast, lstm_model, scaler = fit_lstm(train_data, test_data)
    # metrics_df = evaluate_models(test_data, arima_forecast, lstm_forecast
    # plot_forecasts(train_data, test_data, arima_forecast, lstm_forecast)