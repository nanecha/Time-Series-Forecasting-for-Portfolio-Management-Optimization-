
# Task 1 — Preprocessing & Exploratory Data Analysis

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

