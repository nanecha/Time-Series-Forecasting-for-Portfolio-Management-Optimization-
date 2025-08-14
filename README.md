# Time Series Forecasting for Portfolio Management Optimization

This project focuses on preprocessing, analyzing, forecasting, optimizing, and backtesting a portfolio consisting of **TSLA**, **BND**, and **SPY** using time series data from 2015-07-01 to 2025-07-31. The tasks include data preprocessing, exploratory data analysis (EDA), time series forecasting, portfolio optimization using Modern Portfolio Theory (MPT), and strategic backtesting.

---

## 📂 Folder Structure

```plaintext
project-root/
│
├── data/
│   ├── scrap_data/                 # Raw CSV files fetched via yfinance
│   │   ├── TSLA_data.csv
│   │   ├── SPY_data.csv
│   │   ├── BND_data.csv
│   │   ├── volume_data.csv
│   │   └── adj_close_data.csv
│   │
│   └── output/                     # Cleaned data, metrics, and forecasts
│       ├── adj_close.csv
│       ├── daily_returns.csv
│       ├── rolling_mean.csv
│       ├── rolling_std.csv
│       ├── stationarity_results.csv
│       ├── risk_metrics.csv
│       ├── evaluation_metrics.csv
│       ├── arima_forecast.csv
│       ├── lstm_forecast.csv
│       ├── daily_returns_with_outliers.png
│       ├── adj_close_prices.png
│       ├── daily_returns.png
│       ├── rolling_volatility.png
│       ├── risk_metrics.png
│       ├── arima_forecast.png
│       ├── lstm_forecast.png
│       ├── model_comparison.png
│       ├── efficient_frontier.png
│       ├── backtesting_returns.png
│
├── plots/                          # Visualization outputs
│   └── forecasting/                # Forecast-specific plots
│
├── src/                            # Python scripts with reusable functions
│   ├── __init__.py
│   ├── scrap_preprocess_data.py    # Functions for data fetching, cleaning, EDA
│   └── time_series_forecasting_models.py  # Functions for ARIMA, LSTM, evaluation
│
├── notebooks/
│   ├── preprocessing_expore_data.ipynb              # Task 1: Data preprocessing and EDA
│   ├── forecasting_future_market_trends.ipynb       # Task 2 & 3: Time series forecasting
│   ├── portfolio_optimization.ipynb            # Task 4: Portfolio optimization
│   └── strategy_back_testing.ipynb 
    └── time series forecasting.ipynb                       # Task 5: Strategic backtesting
│
├── requirements.txt                # Python dependencies
├── summary.json                    # Portfolio optimization summary
├── summary.txt                     # Portfolio optimization summary
└── README.md                       # This file

⚙️ Dependencies
Install dependencies from requirements.txt:
bashpip install -r requirements.txt
Main libraries:

yfinance
pandas
numpy
matplotlib
seaborn
statsmodels
pmdarima
scikit-learn
tensorflow


🚀 How to Run

Clone the repository:
bashgit clone https://github.com/yourusername/yourrepo.git
cd yourrepo

Install dependencies:
bashpip install -r requirements.txt

Run the notebooks:

Open the notebooks in notebooks/ using JupyterLab, Jupyter Notebook, or VSCode.
Execute cells sequentially in the following order:

task1_preprocessing_eda.ipynb (Data preprocessing and EDA)
task2_forecasting.ipynb (Time series forecasting for TSLA)
task4_portfolio_optimization.ipynb (Portfolio optimization)
task5_backtesting.ipynb (Strategic backtesting)

Outputs:

Cleaned data and metrics: Saved in data/output/
Plots and visualizations: Saved in plots/ or data/output/
Summary files: summary.json and summary.txt for portfolio optimization results

📊 Workflow Summary
Task 1: Preprocessing & Exploratory Data Analysis

Objective: Prepare and analyze financial time series data for TSLA, BND, and SPY.
Steps:

Fetch historical price and volume data (2015-07-01 to 2025-07-31) via yfinance.
Clean data (handle missing values, set datetime index, ensure correct data types).
Perform EDA: price trends, daily returns, rolling volatility, outlier detection, stationarity tests.
Calculate portfolio risk metrics (Sharpe Ratio, Volatility, VaR).
Save processed datasets and plots in data/output/.



Task 2 & 3: Time Series Forecasting for TSLA

Objective: Forecast TSLA stock prices using ARIMA and LSTM models.
Steps:

Load and preprocess TSLA data from data/scrap_data/TSLA_data.csv.
Split data:

Training: 2015-07-01 to 2023-12-31
Testing: 2024-01-02 to 2025-07-30


Fit ARIMA model using pmdarima.auto_arima.
Fit LSTM model with tensorflow.keras (2 LSTM layers, 50 units, 20% dropout).
Evaluate models using MAE, RMSE, and MAPE.
Save forecasts and evaluation metrics in data/output/ and plots in plots/forecasting/.



Task 4: Portfolio Optimization using MPT

Objective: Optimize portfolio weights for TSLA, BND, and SPY using Modern Portfolio Theory.
Steps:

Load historical prices and TSLA forecast from Task 3.
Compute daily returns and annualized covariance.
Simulate portfolios using Monte Carlo simulation.
Identify portfolios with maximum Sharpe Ratio and minimum volatility.
Plot Efficient Frontier and save in data/output/efficient_frontier.png.
Save summary in summary.json and summary.txt.



Task 5: Strategic Backtesting

Objective: Validate the optimized portfolio strategy by simulating performance.
Steps:

Use the last year of data (2024-08-01 to 2025-07-31) for backtesting.
Define benchmark: 60% SPY / 40% BND static portfolio.
Simulate strategy:

Start with optimal weights from Task 4.
Hold portfolio for one month, then re-run forecast and re-optimize.


Compare cumulative returns of strategy vs. benchmark.
Save performance metrics and plots in data/output/.




📌 Assumptions

Risk-free rate: 2% annually.
Data period: 2015-07-01 to 2025-07-31.
Data source: yfinance for historical OHLCV data.
Directory creation: Data directories (data/, plots/) are created automatically if they don’t exist.
TSLA forecast: Derived from Task 3 (ARIMA or LSTM).
Backtesting period: August 1, 2024 to July 31, 2025.


📈 Key Insights

Task 1 (EDA):

TSLA: High volatility, non-stationary adjusted close, stationary daily returns.
BND: Low volatility, stable, stationary daily returns.
SPY: Moderate volatility, diversified, stationary daily returns.
Risk metrics: TSLA has the highest risk/reward, followed by SPY, then BND.


Task 2 & 3 (Forecasting):

ARIMA outperforms LSTM for TSLA forecasting (lower MAE, RMSE, MAPE).
TSLA’s high volatility leads to higher prediction errors.
Recommendations: Use ARIMA for short-term forecasts, consider log transformation for ARIMA, and tune LSTM hyperparameters.


Task 4 (Portfolio Optimization):

Monte Carlo simulation identifies optimal portfolio weights.
Max Sharpe Ratio portfolio prioritizes higher returns, while min volatility portfolio prioritizes stability.


Task 5 (Backtesting):

Strategy performance is compared against a 60% SPY / 40% BND benchmark.
Re-optimization every month accounts for changing market conditions.




📝 Notes

Ensure src/ is in the Python path for importing helper functions.
LSTM requires MinMaxScaler for data normalization.
ARIMA uses pmdarima.auto_arima for automatic parameter selection.
All CSV files must have correct column names (e.g., 'Date', 'Adj Close').
Adjust file paths and dates based on your dataset.