
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os
import warnings
warnings.filterwarnings('ignore')


def fetch_financial_data(tickers, start_date, end_date, data_dir='F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/scrap data'):
    """
    Fetch financial data for given tickers using YFinance and save to CSV.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols (e.g., ['TSLA', 'BND', 'SPY']).
    start_date : str
        Start date for data in 'YYYY-MM-DD' format.
    end_date : str
        End date for data in 'YYYY-MM-DD' format.
    data_dir : str, optional
        Directory to save CSV data (default: 'data').
    
    Returns:
    --------
    adj_close : DataFrame
        Adjusted Close prices.
    volume : DataFrame
        Volume data.
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    try: 
        # Download data for all tickers at once
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by='ticker'
        )
        if data.empty:
            raise ValueError("No data fetched for the given tickers and date range.")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None

    # Prepare DataFrames for Adjusted Close and Volume
    adj_close = pd.DataFrame()
    volume = pd.DataFrame()
    possible_close_cols = ['Adj Close', 'adj close', 'Adjusted close', 'Close', 'close']

    # Handle case where yfinance returns MultiIndex columns
    for ticker in tickers:
        if ticker not in data.columns.levels[0]:
            print(f"Warning: No data available for {ticker}")
            continue

        # Find the correct "Close" column name
        close_col_found = None
        for col in possible_close_cols:
            if (ticker, col) in data.columns:
                close_col_found = col
                break

        if close_col_found is None:
            print(f"Error: Close or Adjusted Close column not found for {ticker}")
            continue

        # Extract Adjusted Close and Volume data
        adj_close[ticker] = data[(ticker, close_col_found)]
        if (ticker, 'Volume') in data.columns:
            volume[ticker] = data[(ticker, 'Volume')]

        # Save individual ticker data to CSV
        data[ticker].to_csv(f'{data_dir}/{ticker}_data.csv')

    # Save combined data to CSV
    adj_close.to_csv(f'{data_dir}/adj_close_data.csv')
    volume.to_csv(f'{data_dir}/volume_data.csv')

    # Check if any data was successfully fetched
    if adj_close.empty or volume.empty:
        print("Error: No data fetched for the given tickers and date range.")
        return None, None

    print(f"Data fetched successfully for {len(adj_close.columns)} tickers.")
    return adj_close, volume
# %% [markdown]
# ## Cell 4: Define Missing Values Handling Function
# Handle missing values using forward and backward fill.

# %% [code]
def handle_missing_values(adj_close, volume,df):
    """
    Handle missing values in data using forward and backward fill.
    
    Parameters:
    -----------
    adj_close : DataFrame
        Adjusted Close prices.
    volume : DataFrame
        Volume data.
    
    Returns:
    --------
    adj_close : DataFrame
        Cleaned Adjusted Close.
    volume : DataFrame
        Cleaned Volume.
    """
    if adj_close is None or volume is None:
        print("Error: Invalid input data for handling missing values.")
        return None, None
    
    # Handle missing values
    print("\nMissing Values Before Handling:")
    print(adj_close.isna().sum())
    print(volume.isna().sum())
    print(df.isna().sum())
    adj_close = adj_close.fillna(method='ffill').fillna(method='bfill')
    volume = volume.fillna(method='ffill').fillna(method='bfill')
    df = df.fillna(method='ffill').fillna(method='bfill')
    print("\nMissing Values After Handling:")
    print(adj_close.isna().sum())
    print(volume.isna().sum())
    print(df.isna().sum())
    return adj_close, volume, df

# %% [markdown]
# ## Cell 5: Define EDA Function (Visualizations)
# Perform exploratory data analysis with visualizations.

# %% [code]
def eda_data_analysis(adj_close, tickers, output_dir='plots'):
    """
    Perform Exploratory Data Analysis with visualizations.
    
    Parameters:
    -----------
    adj_close : DataFrame
        Adjusted Close prices.
    tickers : list
        List of tickers.
    output_dir : str, optional
        Directory to save plots.
    
    Returns:
    --------
    daily_returns : DataFrame
        Daily percentage returns.
    rolling_mean : DataFrame
        30-day rolling mean of returns.
    rolling_std : DataFrame
        30-day rolling std of returns.
    """
    if adj_close is None:
        print("Error: No Adjusted Close data provided for EDA.")
        return None, None, None
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot Adjusted Close prices
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.plot(adj_close.index, adj_close[ticker], label=ticker)
    plt.title('Adjusted Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price (USD)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/adj_close_prices.png')
    plt.close()
    
    # Calculate and plot daily returns
    daily_returns = adj_close.pct_change().dropna()
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.plot(daily_returns.index, daily_returns[ticker], label=ticker, alpha=0.6)
    plt.title('Daily Percentage Returns')
    plt.xlabel('Date')
    plt.ylabel('Daily Return (%)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/daily_returns.png')
    plt.close()
    
    # Calculate and plot 30-day rolling volatility
    rolling_window = 30
    rolling_mean = daily_returns.rolling(window=rolling_window).mean()
    rolling_std = daily_returns.rolling(window=rolling_window).std()
    
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.plot(rolling_std.index, rolling_std[ticker], label=ticker)
    plt.title('30-Day Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Rolling Standard Deviation')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/rolling_volatility.png')
    plt.close()
    
    return daily_returns, rolling_mean, rolling_std

# %% [markdown]
# ## Cell 6: Define Outlier Detection Function
# Detect outliers in daily returns.

# %% [code]
def outlier_detection(daily_returns, tickers):
    """
    Detect outliers in daily returns.
    
    Parameters:
    -----------
    daily_returns : DataFrame
        Daily percentage returns.
    tickers : list
        List of tickers.
    
    Returns:
    --------
    outliers : dict
        Outliers per ticker.
    means : dict
        Means per ticker.
    stds : dict
        Standard deviations per ticker.
    """
    if daily_returns is None:
        print("Error: No daily returns data provided for outlier detection.")
        return None, None, None
    
    outlier_threshold = 3
    outliers = {}
    means = {}
    stds = {}
    for ticker in tickers:
        means[ticker] = daily_returns[ticker].mean()
        stds[ticker] = daily_returns[ticker].std()
        outliers[ticker] = daily_returns[ticker][abs(daily_returns[ticker] - means[ticker]) > outlier_threshold * stds[ticker]]
    
    print("\nOutliers in Daily Returns (Beyond 3 Std Dev):")
    for ticker in tickers:
        print(f"{ticker}: {len(outliers[ticker])} outliers")
    
    return outliers, means, stds

# %% [markdown]
# ## Cell 7: Define Stationarity Test Function
# Perform Augmented Dickey-Fuller test for stationarity.

# %% [code]
def stationarity_test(adj_close, daily_returns, tickers):
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Parameters:
    -----------
    adj_close : DataFrame
        Adjusted Close prices.
    daily_returns : DataFrame
        Daily percentage returns.
    tickers : list
        List of tickers.
    
    Returns:
    --------
    stationarity_results : dict
        ADF results per ticker.
    """
    if adj_close is None or daily_returns is None:
        print("Error: Invalid input data for stationarity test.")
        return None
    
    stationarity_results = {}
    for ticker in tickers:
        stationarity_results[ticker] = {
            'adj_close': adfuller(adj_close[ticker]),
            'daily_returns': adfuller(daily_returns[ticker].dropna())
        }
    
    print("\nAugmented Dickey-Fuller Test Results:")
    for ticker in tickers:
        print(f"\n{ticker} - Adjusted Close:")
        print(f"ADF Statistic: {stationarity_results[ticker]['adj_close'][0]:.4f}")
        print(f"p-value: {stationarity_results[ticker]['adj_close'][1]:.4f}")
        print("Stationary" if stationarity_results[ticker]['adj_close'][1] < 0.05 else "Non-Stationary")
        
        print(f"\n{ticker} - Daily Returns:")
        print(f"ADF Statistic: {stationarity_results[ticker]['daily_returns'][0]:.4f}")
        print(f"p-value: {stationarity_results[ticker]['daily_returns'][1]:.4f}")
        print("Stationary" if stationarity_results[ticker]['daily_returns'][1] < 0.05 else "Non-Stationary")
    
    return stationarity_results

# %% [markdown]
# ## Cell 8: Define Risk Metrics Function
# Calculate Value at Risk and Sharpe Ratio.

# %% [code]
def risk_metrics(daily_returns, tickers, risk_free_rate=0.02):
    """
    Calculate Value at Risk and Sharpe Ratio.
    
    Parameters:
    -----------
    daily_returns : DataFrame
        Daily percentage returns.
    tickers : list
        List of tickers.
    risk_free_rate : float, optional
        Annual risk-free rate (default: 0.02).
    
    Returns:
    --------
    risk_metrics_df : DataFrame
        VaR and Sharpe Ratio per ticker.
    """
    if daily_returns is None:
        print("Error: No daily returns data provided for risk metrics.")
        return None
    
    # Value at Risk (95% confidence level)
    confidence_level = 0.05
    var = daily_returns.quantile(confidence_level)
    
    print("\n95% Value at Risk (Daily Returns):")
    for ticker in tickers:
        print(f"{ticker}: {var[ticker]:.4f} (A daily loss exceeding this with 5% probability)")
    
    # Sharpe Ratio (annualized)
    daily_risk_free_rate = risk_free_rate / 252
    sharpe_ratio = (daily_returns.mean() - daily_risk_free_rate) / daily_returns.std() * np.sqrt(252)
    
    print("\nSharpe Ratio (Annualized):")
    for ticker in tickers:
        print(f"{ticker}: {sharpe_ratio[ticker]:.4f}")
    
    # Save risk metrics to DataFrame
    risk_metrics_df = pd.DataFrame({
        'VaR': var,
        'Sharpe Ratio': sharpe_ratio
    })
    
    return risk_metrics_df

# %% [markdown]
# ## Cell 9: Define Insights and Documentation Function
# Generate and document dynamic insights.

# %% [code]
def insights_and_documentation(adj_close, daily_returns, rolling_mean, rolling_std, stationarity_results, outliers, risk_metrics_df, output_dir='plots', tickers=None):
    """
    Generate and document dynamic insights.
    
    Parameters:
    -----------
    adj_close : DataFrame
        Adjusted Close prices.
    daily_returns : DataFrame
        Daily percentage returns.
    rolling_mean : DataFrame
        Rolling mean.
    rolling_std : DataFrame
        Rolling std.
    stationarity_results : dict
        Stationarity test results.
    outliers : dict
        Outliers per ticker.
    risk_metrics_df : DataFrame
        Risk metrics.
    output_dir : str, optional
        Directory to save insights.
    tickers : list
        List of tickers.
    
    Returns:
    --------
    results : dict
        Compiled results including insights.
    """
    if adj_close is None or daily_returns is None or stationarity_results is None or risk_metrics_df is None:
        print("Error: Invalid input data for generating insights.")
        return None
    
    if tickers is None:
        tickers = adj_close.columns
    
    # Dynamic Insights
    insights = []
    asset_descriptions = {
        'TSLA': "High-growth, high-risk stock in the consumer discretionary sector (Automobile Manufacturing).",
        'BND': "A bond ETF tracking U.S. investment-grade bonds, providing stability and income.",
        'SPY': "An ETF tracking the S&P 500 Index, offering broad U.S. market exposure."
    }
    
    for ticker in tickers:
        is_adj_stationary = "Stationary" if stationarity_results[ticker]['adj_close'][1] < 0.05 else "Non-Stationary"
        is_returns_stationary = "Stationary" if stationarity_results[ticker]['daily_returns'][1] < 0.05 else "Non-Stationary"
        num_outliers = len(outliers[ticker])
        var_value = risk_metrics_df.loc[ticker, 'VaR']
        sharpe_value = risk_metrics_df.loc[ticker, 'Sharpe Ratio']
        
        insights.append(f"{ticker}: {asset_descriptions.get(ticker, 'Unknown')}.")
        insights.append(f"{ticker}: Adjusted Close is {is_adj_stationary} (p-value: {stationarity_results[ticker]['adj_close'][1]:.4f}).")
        insights.append(f"{ticker}: Daily Returns are {is_returns_stationary} (p-value: {stationarity_results[ticker]['daily_returns'][1]:.4f}).")
        insights.append(f"{ticker}: Number of outliers in daily returns: {num_outliers}.")
        insights.append(f"{ticker}: 95% VaR: {var_value:.4f} (potential daily loss at 5% probability).")
        insights.append(f"{ticker}: Sharpe Ratio (annualized): {sharpe_value:.4f}.")
    
    # Add general insights
    insights.append("Overall: TSLA shows high volatility, BND provides stability, SPY offers moderate risk.")
    insights.append("Stationarity Implications: Non-stationary series may require differencing for ARIMA modeling.")
    
    # Save insights to a text file
    with open(f'{output_dir}/eda_insights.txt', 'w') as f:
        for insight in insights:
            f.write(insight + '\n')
    
    # Return results
    return {
        'adj_close': adj_close,
        'daily_returns': daily_returns,
        'rolling_stats': {'mean': rolling_mean, 'std': rolling_std},
        'risk_metrics': risk_metrics_df,
        'stationarity_results': stationarity_results,
        'outliers': outliers,
        'insights': insights
    }
if __name__ == "__main__":
    pass
