# task-4
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 0) Config
# -----------------------------
CONFIG = {
    # Paths to historical CSVs (from Task 1)
    "tsla_hist_csv": "F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/scrap data/TSLA_data.csv",
    "bnd_hist_csv":  "F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/scrap data/BND_data.csv",
    "spy_hist_csv":  "F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/scrap data/SPY_data.csv",

    # Path to TSLA forecast results (from Task 3)
    "tsla_forecast_csv": "F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/output/data/output/tesla_forecast_6months.csv",  # columns: Forecast, Lower, Upper

    # Columns
    "date_col": "Date",
    "price_col": "Close",  # fallback to 'Close' if not present

    # Output
    "output_dir": "data/output",
    "frontier_png": "efficient_frontier.png",
    "frontier_csv": "efficient_frontier_samples.csv",
    "summary_json": "portfolio_summary.json",
    "summary_txt": "portfolio_summary.txt",

    # Simulation
    "n_portfolios": 50000,
    "risk_free_rate": 0.02,  # annualized
    "trading_days": 252
}

# -----------------------------
# 1) Data loaders
# -----------------------------
def _load_price_csv(path, date_col="Date", price_col="Close"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"{date_col} column missing in {path}")

    # Normalize column names
    candidates = [price_col, "Adjusted Close", "AdjClose", "Close", "close"]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise ValueError(f"No price column found in {path}. Tried: {candidates}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    series = df[col].astype(float).rename(os.path.basename(path).split("_")[0])  # name by ticker prefix
    return series

def load_historical_prices(cfg=CONFIG):
    tsla = _load_price_csv(cfg["tsla_hist_csv"], cfg["date_col"], cfg["price_col"]).rename("TSLA")
    bnd  = _load_price_csv(cfg["bnd_hist_csv"],  cfg["date_col"], cfg["price_col"]).rename("BND")
    spy  = _load_price_csv(cfg["spy_hist_csv"],  cfg["date_col"], cfg["price_col"]).rename("SPY")

    # Align by intersection of dates
    prices = pd.concat([tsla, bnd, spy], axis=1).dropna()
    return prices

def load_tsla_expected_return_from_forecast(cfg=CONFIG):
    """
    Convert the TSLA 6-month forecast path to an expected annual return.
    Approach:
      - Compute average daily forecasted return over forecast horizon relative to its previous day.
      - Annualize via 252 trading days: (1 + mean_daily)^252 - 1
    Assumes forecast CSV has columns: ['Forecast','Lower','Upper'] and date index or a 'Date' column.
    """
    path = cfg["tsla_forecast_csv"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"TSLA forecast file not found: {path}")

    df = pd.read_csv(path)
    # Detect date column or index
    if cfg["date_col"] in df.columns:
        df[cfg["date_col"]] = pd.to_datetime(df[cfg["date_col"]])
        df = df.set_index(cfg["date_col"])

    if "Forecast" not in df.columns:
        raise ValueError("Forecast column missing in TSLA forecast CSV. Expected columns: ['Forecast','Lower','Upper'].")

    forecast = df["Forecast"].astype(float)
    # Daily returns inside the forecast horizon
    daily_ret = forecast.pct_change().dropna()
    if daily_ret.empty:
        # Fallback: use last forecast against first forecast (coarse)
        if len(forecast) >= 2:
            daily_ret = pd.Series([(forecast.iloc[-1] / forecast.iloc[0]) ** (1/len(forecast)) - 1])
        else:
            raise ValueError("Not enough forecast points to compute expected return.")

    mean_daily = daily_ret.mean()
    ann_expected = (1 + mean_daily) ** cfg["trading_days"] - 1
    return float(ann_expected)

# -----------------------------
# 2) Return stats
# -----------------------------
def compute_daily_returns(prices: pd.DataFrame):
    returns = prices.pct_change().dropna()
    return returns

def expected_returns_annual(returns: pd.DataFrame, tsla_expected_ann: float, cfg=CONFIG):
    """
    Expected returns:
      - TSLA: from forecast (annual)
      - BND/SPY: historical mean daily * 252
    """
    mean_daily = returns.mean()
    exp_ann = mean_daily * cfg["trading_days"]
    exp_ann["TSLA"] = tsla_expected_ann
    return exp_ann

def covariance_matrix_annual(returns: pd.DataFrame, cfg=CONFIG):
    # Daily covariance * 252 -> annualized covariance
    return returns.cov() * cfg["trading_days"]

# -----------------------------
# 3) Monte Carlo Efficient Frontier
# -----------------------------
def simulate_portfolios(exp_returns, cov_matrix, rf=0.02, n=50000):
    """
    Random weight simulation subject to sum(weights)=1, weights>=0
    """
    tickers = list(exp_returns.index)
    results = []
    weights_list = []

    chol = np.linalg.cholesky(cov_matrix.values)  # for speed sometimes, but not strictly required

    for _ in range(n):
        w = np.random.random(len(tickers))
        w /= w.sum()

        port_return = float(np.dot(w, exp_returns.values))
        port_vol = float(np.sqrt(np.dot(w, np.dot(cov_matrix.values, w))))
        sharpe = (port_return - rf) / (port_vol + 1e-12)

        results.append([port_vol, port_return, sharpe])
        weights_list.append(w)

    results = np.array(results)
    df = pd.DataFrame(results, columns=["Volatility", "Return", "Sharpe"])
    wdf = pd.DataFrame(weights_list, columns=tickers)
    out = pd.concat([df, wdf], axis=1)
    return out

def pick_key_portfolios(frontier_df, tickers=("TSLA","BND","SPY")):
    i_max_sharpe = frontier_df["Sharpe"].idxmax()
    i_min_vol    = frontier_df["Volatility"].idxmin()

    max_sharpe = frontier_df.loc[i_max_sharpe, ["Volatility","Return","Sharpe", *tickers]].to_dict()
    min_vol    = frontier_df.loc[i_min_vol,    ["Volatility","Return","Sharpe", *tickers]].to_dict()
    return max_sharpe, min_vol

# -----------------------------
# 4) Plot
# -----------------------------
def plot_efficient_frontier(frontier_df, max_sharpe, min_vol, cfg=CONFIG):
    plt.figure(figsize=(10,6))
    plt.scatter(frontier_df["Volatility"], frontier_df["Return"], s=6, alpha=0.35, label="Portfolios")

    # Mark key portfolios
    plt.scatter(max_sharpe["Volatility"], max_sharpe["Return"], marker="*", s=220, label="Max Sharpe", edgecolor="black")
    plt.scatter(min_vol["Volatility"],    min_vol["Return"],    marker="X", s=120, label="Min Volatility", edgecolor="black")

    plt.xlabel("Annualized Volatility (Risk)")
    plt.ylabel("Annualized Expected Return")
    plt.title("Efficient Frontier — TSLA / BND / SPY")
    plt.legend()
    plt.grid(True)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], cfg["frontier_png"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    return out_path

# -----------------------------
# 5) Recommend Portfolio
# -----------------------------
def recommend_portfolio(max_sharpe, min_vol, preference="max_sharpe"):
    """
    preference: "max_sharpe" or "min_vol"
    """
    return max_sharpe if preference == "max_sharpe" else min_vol

def write_summary(exp_returns, cov_matrix, max_sharpe, min_vol, recommended, cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)

    summary = {
        "assumptions": {
            "risk_free_rate": cfg["risk_free_rate"],
            "trading_days": cfg["trading_days"],
            "notes": [
                "TSLA expected return from forecast (annualized).",
                "BND, SPY expected returns from historical mean daily returns (annualized).",
                "Covariance matrix from historical daily returns, annualized.",
            ]
        },
        "expected_returns_annual": exp_returns.to_dict(),
        "cov_matrix_annual": cov_matrix.round(6).to_dict(),
        "max_sharpe_portfolio": max_sharpe,
        "min_vol_portfolio": min_vol,
        "recommended": recommended,
        "recommendation_rationale": (
            "Max Sharpe portfolio selected for highest risk-adjusted return."
            if recommended is max_sharpe else
            "Min Volatility portfolio selected to prioritize lower risk."
        ),
    }

    # JSON
    json_path = os.path.join(cfg["output_dir"], cfg["summary_json"])
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # TXT
    txt_path = os.path.join(cfg["output_dir"], cfg["summary_txt"])
    lines = []
    lines.append("=== Task 4: Portfolio Optimization Summary ===")
    lines.append("")
    lines.append("Expected Annual Returns:")
    for k, v in exp_returns.items():
        lines.append(f"  - {k}: {v:.4%}")
    lines.append("")
    lines.append("Max Sharpe Portfolio:")
    lines.append(f"  Return:   {max_sharpe['Return']:.4%}")
    lines.append(f"  Volatility: {max_sharpe['Volatility']:.4%}")
    lines.append(f"  Sharpe:   {max_sharpe['Sharpe']:.3f}")
    for k in ["TSLA","BND","SPY"]:
        lines.append(f"  w_{k}:    {max_sharpe[k]:.2%}")
    lines.append("")
    lines.append("Min Volatility Portfolio:")
    lines.append(f"  Return:   {min_vol['Return']:.4%}")
    lines.append(f"  Volatility: {min_vol['Volatility']:.4%}")
    lines.append(f"  Sharpe:   {min_vol['Sharpe']:.3f}")
    for k in ["TSLA","BND","SPY"]:
        lines.append(f"  w_{k}:    {min_vol[k]:.2%}")
    lines.append("")
    lines.append("Recommended Portfolio:")
    lines.append(f"  Choice: {'Max Sharpe (Tangency)' if recommended is max_sharpe else 'Min Volatility'}")
    lines.append(f"  Return:   {recommended['Return']:.4%}")
    lines.append(f"  Volatility: {recommended['Volatility']:.4%}")
    lines.append(f"  Sharpe:   {recommended['Sharpe']:.3f}")
    for k in ["TSLA","BND","SPY"]:
        lines.append(f"  w_{k}:    {recommended[k]:.2%}")
    lines.append("")
    lines.append("Rationale: " + (
        "Prioritizing maximum risk-adjusted return (Sharpe). Suitable when investor can tolerate moderate risk."
        if recommended is max_sharpe else
        "Prioritizing minimal risk. Suitable when capital preservation and lower volatility are key."
    ))

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    return json_path, txt_path


if __name__ == "__main__":
    # preference options: "max_sharpe" or "min_vol"
    pass
