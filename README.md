
# Time Series Forecasting for Portfolio Management Optimization

## Task 1 — Preprocessing & Exploratory Data Analysis

This task is part of the **GMF Investments Case Study**, where we prepare and analyze financial time series data for **TSLA**, **BND**, and **SPY**.

The notebook:

* Fetches historical price & volume data (2015-07-01 → 2025-07-31) via [`yfinance`](https://pypi.org/project/yfinance/).
* Cleans and preprocesses data (handles missing values, ensures correct data types, manages multi-index columns).
* Performs Exploratory Data Analysis (EDA) — price trends, returns, volatility, outlier detection, seasonality, and stationarity tests.
* Calculates portfolio risk metrics (Sharpe Ratio, Volatility, etc.).
* Saves processed datasets and plots for later modeling.

---

## 📂 Folder Structure

```plaintext
project-root/
│
├── data/
│   ├── scrap data/                 # Raw CSV files fetched via yfinance
│   │   ├── TSLA_data.csv
│   │   ├── SPY_data.csv
│   │   ├── BND_data.csv
│   │   ├── volume_data.csv
│   │   └── adj_close_data.csv
│   │
│   └── output/                     # Cleaned data & computed metrics
│       ├── adj_close.csv
│       ├── daily_returns.csv
│       ├── rolling_mean.csv
│       ├── rolling_std.csv
│       ├── stationarity_results.csv
│       ├── risk_metrics.csv
│       ├── daily_returns_with_outliers.png
│       ├── adj_close_prices.png
│       ├── daily_returns.png
│       ├── rolling_volatility.png
│       ├── risk_metrics.png
│       └── ... other plots ...
│
├── plots/                          # (Optional) Visualization outputs
│
├── src/                            # Python scripts with reusable functions
│   ├── __init__.py
│   └── scrap_preprocess_data.py    # All helper functions: fetching, cleaning, EDA, metrics
│
├── notebooks/
│   └── task1_preprocessing_eda.ipynb  # Main Jupyter Notebook for Task 1
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## ⚙️ Dependencies

Install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Main libraries:**

* `yfinance`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `statsmodels`

---

## 🚀 How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the notebook**:

   * Open `notebooks/task1_preprocessing_eda.ipynb` in JupyterLab, Jupyter Notebook, or VSCode.
   * Execute cells sequentially.

4. **Outputs**:

   * Cleaned CSV files → `data/output/`
   * Plots & visualizations → `plots/` or `data/output/`
   * Metrics & insights stored in the `results` dictionary for later modeling.

---

## 📊 Workflow Summary

**Step 1 — Data Fetching**

* Fetch Adjusted Close & Volume data for given tickers via `fetch_financial_data`.

**Step 2 — Data Cleaning**

* Handle missing values via forward/backward fill.
* Convert date columns to `datetime` and set index.

**Step 3 — Exploratory Data Analysis**

* Price trends (`adj_close_prices.png`)
* Daily returns (`daily_returns.png`)
* Rolling volatility (`rolling_volatility.png`)
* Outlier detection (`daily_returns_with_outliers.png`)
* Stationarity test (`stationarity_results.csv`)

**Step 4 — Risk Metrics**

* Compute Sharpe Ratio, Volatility, etc. (`risk_metrics.csv` + plot).

**Step 5 — Save Outputs**

* All cleaned data, metrics, and plots stored for next stages.

---

## 📌 Assumptions

* **Risk-free rate**: 2% annually.
* **Period**: 2015-07-01 to 2025-07-31.
* All scripts in `src/` are imported by the notebook.
* Data directories exist or will be created automatically.

---

---
# Financial Risk & Performance Summary

## TSLA (Tesla, Inc.)
- **Sector:** Consumer Discretionary — Automobile Manufacturing  
- **Profile:** High-growth, high-risk stock  
- **Stationarity Tests:**
  - Adjusted Close: **Non-Stationary** (p-value: 0.5732)
  - Daily Returns: **Stationary** (p-value: 0.0000)
- **Risk Metrics:**
  - Outliers in Daily Returns: **41**
  - 95% Value-at-Risk (VaR): **-0.0547** (potential daily loss at 5% probability)
- **Performance:**
  - Sharpe Ratio (annualized): **0.7446**

---

## BND (Vanguard Total Bond Market ETF)
- **Sector:** Fixed Income — U.S. Investment-Grade Bonds  
- **Profile:** Stability and income provider  
- **Stationarity Tests:**
  - Adjusted Close: **Non-Stationary** (p-value: 0.5155)
  - Daily Returns: **Stationary** (p-value: 0.0000)
- **Risk Metrics:**
  - Outliers in Daily Returns: **26**
  - 95% VaR: **-0.0049** (potential daily loss at 5% probability)
- **Performance:**
  - Sharpe Ratio (annualized): **-0.0073**

---

## SPY (SPDR S&P 500 ETF)
- **Sector:** Equities — Broad U.S. Market Exposure  
- **Profile:** Moderate risk, diversified  
- **Stationarity Tests:**
  - Adjusted Close: **Non-Stationary** (p-value: 0.9897)
  - Daily Returns: **Stationary** (p-value: 0.0000)
- **Risk Metrics:**
  - Outliers in Daily Returns: **35**
  - 95% VaR: **-0.0172** (potential daily loss at 5% probability)
- **Performance:**
  - Sharpe Ratio (annualized): **0.6844**

---

## Overall Insights
- **Volatility:** TSLA > SPY > BND  
- **Risk Profile:** TSLA is high-risk/high-reward, BND is stable, SPY is balanced.  
- **Stationarity Implication:** All Adjusted Close series are non-stationary, suggesting a need for differencing before ARIMA or similar time series modeling.  
- **Daily Returns:** All are stationary, making them suitable for risk modeling without further differencing.

---

# Task 2: TSLA Price Forecasting

## 📌 Introduction
Task 2 of the GMF Investments case study focuses on forecasting **Tesla (TSLA) Adjusted Close** prices to support portfolio management decisions. Two models were implemented:

- **AutoRegressive Integrated Moving Average (ARIMA)**
- **Long Short-Term Memory (LSTM) Neural Network**

The dataset was split chronologically into:
- **Training:** July 1, 2015 – December 31, 2023 (**2140 observations**)
- **Testing:** January 2, 2024 – July 30, 2025 (**395 observations**)

Model performance was evaluated using **MAE**, **RMSE**, and **MAPE**. Results were visualized in plots and saved as CSV files. This report summarizes the **methodology**, **performance**, **insights**, and **recommendations** for Task 3 (portfolio optimization).

---

## 📂 Methodology

### 1. Data Preparation
- **Source:** `data/TSLA_data.csv` via `yfinance`
- **Cleaning:** Forward and backward fill for missing values
- **Split:**
  - Training: `2015-07-01` → `2023-12-31`
  - Testing: `2024-01-02` → `2025-07-30`  
- Test period capped to avoid forecasting beyond available data as of `2025-08-13`

---

### 2. ARIMA Model
- **Library:** `pmdarima` (`auto_arima`)
- **Best model:** `SARIMAX(0,1,0)(2,0,0)[12]`
- **AIC:** `13622.971`
- **Key Points:**
  - Non-seasonal: `(0,1,0)` → random walk with first-order differencing
  - Seasonal: `(2,0,0)[12]` → annual seasonality
  - **Diagnostics:**  
    - Ljung-Box Q: `p=0.19` → no significant autocorrelation  
    - Jarque-Bera: `p=0.00`, kurtosis = `14.40` → non-normal residuals due to volatility
- **Forecast Horizon:** 395 days (aligned with test period)

---

### 3. LSTM Model
- **Library:** `tensorflow.keras`
- **Architecture:**
  - 2× LSTM layers (50 units)
  - Dropout layers (20%)
  - Dense output layer
- **Training Setup:**
  - Sequence length: `60 days`
  - Epochs: `50`
  - Batch size: `32`
  - Optimizer: Adam (lr = `0.001`)
  - Loss: MSE
- **Scaling:** `MinMaxScaler` to [0, 1]
- **Validation Loss:** `5.6341e-04` (epoch 48)

---

### 4. Evaluation
- **Metrics:** MAE, RMSE, MAPE
- **Results saved to:** `data/output/evaluation_metrics.csv`
- **Plots stored in:** `plots/forecasting/`

---

## 📊 Results

| Model | MAE       | RMSE      | MAPE (%)  |
|-------|-----------|-----------|-----------|
| ARIMA | 63.036976 | 77.656542 | 24.299210 |
| LSTM  | 178.761717| 223.053794| 59.882682 |

- **ARIMA Forecast:** `plots/forecasting/arima_forecast.png`  
- **LSTM Forecast:** `plots/forecasting/lstm_forecast.png`  
- **Model Comparison:** `plots/forecasting/model_comparison.png`  

---

## 🔍 Insights
1. **ARIMA Outperforms LSTM** – lower MAE, RMSE, and MAPE.
2. **High Volatility Impact** – both models have relatively high MAPE due to TSLA’s volatility.
3. **Simple Seasonal Structure Works** – SARIMAX captures trend + annual fluctuations.
4. **LSTM Overfitting** – validation loss improved, but test performance degraded.
5. **Error Metrics** – RMSE > MAE for both models due to large price spikes.

---

## ✅ Recommendations
1. **Use ARIMA for Short-Term Forecasts** – more stable for TSLA’s recent trends.
2. **Log Transformation for ARIMA** – may improve residual normality and reduce MAPE.
3. **Improve LSTM** – increase sequence length, tune hyperparameters, and add more layers to capture long-term dependencies.

---

