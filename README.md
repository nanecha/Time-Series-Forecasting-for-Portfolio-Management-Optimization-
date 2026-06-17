# Time Series Forecasting for Portfolio Management Optimization

This project analyzes and optimizes a portfolio of **TSLA, BND, and SPY** using historical market data (2015–2025). It covers data preprocessing, exploratory analysis, stock price forecasting (ARIMA & LSTM), portfolio optimization using **Modern Portfolio Theory (MPT)**, and strategy backtesting.

## Project Structure

- **data/** – Raw and processed datasets
- **src/** – Reusable Python scripts
- **notebooks/** – Analysis and forecasting workflows
- **plots/** – Generated visualizations
- **summary.json / summary.txt** – Portfolio optimization results
- **requirements.txt** – Project dependencies

## Key Tasks

### 1. Data Preprocessing & EDA
- Download TSLA, BND, and SPY data via `yfinance`
- Clean and transform datasets
- Analyze trends, returns, volatility, and stationarity
- Calculate risk metrics (Sharpe Ratio, VaR, Volatility)

### 2. Time Series Forecasting
- Forecast TSLA prices using:
  - **ARIMA**
  - **LSTM**
- Evaluate performance using MAE, RMSE, and MAPE

### 3. Portfolio Optimization
- Apply **Modern Portfolio Theory (MPT)**
- Generate Efficient Frontier
- Identify:
  - Maximum Sharpe Ratio Portfolio
  - Minimum Volatility Portfolio

### 4. Strategy Backtesting
- Test optimized portfolio performance
- Compare against a **60% SPY / 40% BND** benchmark
- Rebalance monthly based on updated forecasts

## Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Statsmodels, pmdarima
- Scikit-learn
- TensorFlow
- yfinance

## Results
- ARIMA achieved better forecasting accuracy than LSTM for TSLA.
- TSLA showed the highest risk and return potential.
- Portfolio optimization improved risk-adjusted returns.
- Dynamic rebalancing outperformed the benchmark in selected market conditions.

## Installation

```bash
pip install -r requirements.txt