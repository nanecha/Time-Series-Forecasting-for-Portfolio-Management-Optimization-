```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(tsla_path, bnd_path, spy_path, portfolio_summary_path, start_date='2024-08-01', end_date='2025-07-31'):
    """
    Load historical data and Task 4 portfolio weights.
    
    Parameters:
    -----------
    tsla_path, bnd_path, spy_path : str
        Paths to TSLA_data.csv, BND_data.csv, SPY_data.csv.
    portfolio_summary_path : str
        Path to task4_portfolio_summary.txt.
    start_date, end_date : str
        Backtesting period (YYYY-MM-DD).
    
    Returns:
    --------
    returns : DataFrame
        Daily returns for TSLA, BND, SPY.
    strategy_weights : array
        Optimal weights from Task 4 [TSLA, BND, SPY].
    benchmark_weights : array
        Benchmark weights [0, 0.4, 0.6].
    """
    try:
        # Load historical data
        assets = ['TSLA', 'BND', 'SPY']
        data_paths = [tsla_path, bnd_path, spy_path]
        historical_data = pd.DataFrame()
        
        for asset, path in zip(assets, data_paths):
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            historical_data[asset] = df['Adj Close'][start_date:end_date].fillna(method='ffill').fillna(method='bfill')
        
        # Compute daily returns
        returns = historical_data.pct_change().dropna()
        
        # Load Task 4 weights
        with open(portfolio_summary_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('Weights:'):
                    weights_lines = lines[lines.index(line)+1:lines.index(line)+4]
                    strategy_weights = np.array([float(line.split(':')[1].split('(')[0]) for line in weights_lines])
                    break
        else:
            raise ValueError("Weights not found in task4_portfolio_summary.txt")
        
        # Define benchmark weights (0% TSLA, 40% BND, 60% SPY)
        benchmark_weights = np.array([0.0, 0.4, 0.6])
        
        # Validate data
        if returns.empty:
            raise ValueError("No data available for backtesting period.")
        if returns.isna().any().any():
            raise ValueError("NaN values in returns.")
        if not np.isclose(strategy_weights.sum(), 1.0) or not np.isclose(benchmark_weights.sum(), 1.0):
            raise ValueError("Portfolio weights do not sum to 1.")
        
        print(f"Backtesting period: {returns.index.min()} to {returns.index.max()}")
        print(f"Strategy Weights: {dict(zip(assets, strategy_weights))}")
        print(f"Benchmark Weights: {dict(zip(assets, benchmark_weights))}")
        
        return returns, strategy_weights, benchmark_weights
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def simulate_portfolio(returns, weights):
    """
    Simulate portfolio performance.
    
    Parameters:
    -----------
    returns : DataFrame
        Daily returns for assets.
    weights : array
        Portfolio weights.
    
    Returns:
    --------
    portfolio_returns : Series
        Daily portfolio returns.
    cumulative_returns : Series
        Cumulative portfolio returns.
    """
    portfolio_returns = (returns * weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    return portfolio_returns, cumulative_returns

def calculate_metrics(portfolio_returns, risk_free_rate=0.03/252):
    """
    Calculate annualized Sharpe Ratio and total return.
    
    Parameters:
    -----------
    portfolio_returns : Series
        Daily portfolio returns.
    risk_free_rate : float
        Daily risk-free rate.
    
    Returns:
    --------
    sharpe_ratio : float
        Annualized Sharpe Ratio.
    total_return : float
        Total return over the period.
    """
    annualized_return = portfolio_returns.mean() * 252
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - (risk_free_rate * 252)) / annualized_volatility
    total_return = (1 + portfolio_returns).prod() - 1
    return sharpe_ratio, total_return

def plot_cumulative_returns(strategy_cumulative, benchmark_cumulative, output_dir='plots/backtesting'):
    """
    Plot cumulative returns for strategy and benchmark.
    
    Parameters:
    -----------
    strategy_cumulative : Series
        Strategy portfolio cumulative returns.
    benchmark_cumulative : Series
        Benchmark portfolio cumulative returns.
    output_dir : str
        Directory to save plot.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(strategy_cumulative.index, strategy_cumulative * 100, label='Strategy Portfolio', color='blue')
        plt.plot(benchmark_cumulative.index, benchmark_cumulative * 100, label='Benchmark (60% SPY, 40% BND)', color='green')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.title('Strategy vs. Benchmark Cumulative Returns (Aug 2024 - Jul 2025)')
        plt.legend()
        plt.grid()
        plt.savefig(f'{output_dir}/task5_cumulative_returns.png')
        plt.close()
        
        print(f"Cumulative returns plot saved to {output_dir}/task5_cumulative_returns.png")
    except Exception as e:
        print(f"Error plotting cumulative returns: {e}")

def summarize_performance(strategy_metrics, benchmark_metrics, output_dir='data/output'):
    """
    Summarize backtesting results and conclude.
    
    Parameters:
    -----------
    strategy_metrics : tuple
        Strategy portfolio (Sharpe Ratio, total return).
    benchmark_metrics : tuple
        Benchmark portfolio (Sharpe Ratio, total return).
    output_dir : str
        Directory to save summary.
    
    Returns:
    --------
    summary : str
        Backtesting summary and conclusion.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        strategy_sharpe, strategy_return = strategy_metrics
        benchmark_sharpe, benchmark_return = benchmark_metrics
        
        # Conclusion
        outperforms = strategy_return > benchmark_return and strategy_sharpe > benchmark_sharpe
        conclusion = (
            f"The strategy portfolio {'outperforms' if outperforms else 'underperforms'} the benchmark. "
            f"{'Higher total return and Sharpe Ratio indicate the model-driven approach effectively leverages TSLA’s forecast.' if outperforms else 'The benchmark’s stability may be preferable for risk-averse investors.'} "
            "This suggests the forecasting and optimization approach is viable but may require dynamic rebalancing to improve performance."
        )
        
        summary = "\n".join([
            "Task 5: Strategy Backtesting Summary",
            "=" * 40,
            "Backtesting Period: August 1, 2024 - July 31, 2025",
            "Strategy Portfolio:",
            f"  Total Return: {strategy_return:.4f} ({strategy_return*100:.2f}%)",
            f"  Annualized Sharpe Ratio: {strategy_sharpe:.4f}",
            "Benchmark Portfolio (60% SPY, 40% BND):",
            f"  Total Return: {benchmark_return:.4f} ({benchmark_return*100:.2f}%)",
            f"  Annualized Sharpe Ratio: {benchmark_sharpe:.4f}",
            "Conclusion:",
            conclusion
        ])
        
        with open(f'{output_dir}/task5_backtesting_summary.txt', 'w') as f:
            f.write(summary)
        
        print("Backtesting summary saved to data/output/task5_backtesting_summary.txt")
        print("\n" + summary)
        
        return summary
    except Exception as e:
        print(f"Error summarizing performance: {e}")
        return None

def main():
    """
    Main function to execute Task 5 backtesting.
    """
    # Parameters
    tsla_path = 'F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/TSLA_data.csv'
    bnd_path = 'F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/BND_data.csv'
    spy_path = 'F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/SPY_data.csv'
    portfolio_summary_path = 'F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/output/task4_portfolio_summary.txt'
    start_date = '2024-08-01'
    end_date = '2025-07-31'
    output_dir = 'data/output'
    plot_dir = 'plots/backtesting'
    risk_free_rate = 0.03 / 252  # Daily risk-free rate
    
    # Load data and weights
    returns, strategy_weights, benchmark_weights = load_data(tsla_path, bnd_path, spy_path, portfolio_summary_path, start_date, end_date)
    if returns is None:
        return
    
    # Simulate portfolios
    strategy_returns, strategy_cumulative = simulate_portfolio(returns, strategy_weights)
    benchmark_returns, benchmark_cumulative = simulate_portfolio(returns, benchmark_weights)
    
    # Calculate metrics
    strategy_metrics = calculate_metrics(strategy_returns, risk_free_rate)
    benchmark_metrics = calculate_metrics(benchmark_returns, risk_free_rate)
    
    # Plot cumulative returns
    plot_cumulative_returns(strategy_cumulative, benchmark_cumulative, plot_dir)
    
    # Summarize performance
    summarize_performance(strategy_metrics, benchmark_metrics, output_dir)

if __name__ == "__main__":
    main()
```