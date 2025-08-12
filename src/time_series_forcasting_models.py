# src/time_series_forcasting_models.py
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


def prepare_data(data_path, train_end='2023-12-31', test_start='2024-01-01'):
    """
    Load and split TSLA data into training and testing sets.
    
    Parameters:
    -----------
    data_path : str
        Path to TSLA_data.csv.
    train_end : str
        End date for training data (YYYY-MM-DD).
    test_start : str
        Start date for testing data (YYYY-MM-DD).
    
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
        adj_close = df['Adj Close']
        
        # Handle missing values
        adj_close = adj_close.fillna(method='ffill').fillna(method='bfill')
        
        train_data = adj_close[:train_end]
        test_data = adj_close[test_start:]
        
        if train_data.empty or test_data.empty:
            raise ValueError("Training or testing data is empty. Check date range or data availability.")
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Testing data shape: {test_data.shape}")
        
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
        Testing data (Adjusted Close prices, for index alignment).
    output_dir : str
        Directory to save forecast CSV.
    
    Returns:
    --------
    arima_forecast : Series
        Forecasted values with test index.
    arima_model : ARIMA
        Fitted ARIMA model.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        arima_model = auto_arima(train_data, start_p=0, start_q=0, max_p=5, max_d=2, max_q=5,
                                 seasonal=False, stepwise=True, trace=True, error_action='ignore',
                                 suppress_warnings=True)
        arima_forecast = arima_model.predict(n_periods=len(test_data))
        arima_forecast = pd.Series(arima_forecast, index=test_data.index)
        
        if arima_forecast.isna().any():
            raise ValueError("NaN values found in ARIMA forecast.")
        
        arima_forecast.to_csv(f'{output_dir}/arima_forecast.csv')
        return arima_forecast, arima_model
    except Exception as e:
        print(f"Error fitting ARIMA: {e}")
        return None, None


def fit_lstm(train_data, test_data, seq_length=60, epochs=50, batch_size=32, output_dir='data/output'):
    """
    Fit LSTM model and forecast.
    
    Parameters:
    -----------
    train_data : Series
        Training data (Adjusted Close prices).
    test_data : Series
        Testing data (Adjusted Close prices).
    seq_length : int
        Lookback window for sequences.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    output_dir : str
        Directory to save forecast CSV.
    
    Returns:
    --------
    lstm_forecast : Series
        Forecasted values with test index.
    lstm_model : Sequential
        Fitted LSTM model.
    scaler : MinMaxScaler
        Fitted scaler.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Scale data
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
        test_scaled = scaler.transform(test_data.values.reshape(-1, 1))
        
        # Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_sequences(train_scaled, seq_length)
        
        # Build LSTM model
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train LSTM
        lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
        
        # Forecast
        lstm_predictions = []
        current_batch = train_scaled[-seq_length:].reshape(1, seq_length, 1)
        
        for i in range(len(test_data)):
            pred = lstm_model.predict(current_batch, verbose=0)
            if np.isnan(pred).any():
                raise ValueError("NaN values in LSTM predictions.")
            lstm_predictions.append(pred[0, 0])
            current_batch = np.roll(current_batch, -1, axis=1)
            current_batch[0, -1, 0] = pred[0, 0]
        
        # Inverse transform
        lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()
        lstm_forecast = pd.Series(lstm_predictions, index=test_data.index[:len(lstm_predictions)])
        
        if lstm_forecast.isna().any():
            raise ValueError("NaN values found in LSTM forecast.")
        
        lstm_forecast.to_csv(f'{output_dir}/lstm_forecast.csv')
        return lstm_forecast, lstm_model, scaler
    except Exception as e:
        print(f"Error fitting LSTM: {e}")
        return None, None, None


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
        
        # Calculate metrics
        arima_mae = mean_absolute_error(test_data, arima_forecast)
        arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
        arima_mape = np.mean(np.abs((test_data - arima_forecast) / test_data)) * 100
        
        lstm_mae = mean_absolute_error(test_data, lstm_forecast)
        lstm_rmse = np.sqrt(mean_squared_error(test_data, lstm_forecast))
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
        plt.sow()
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