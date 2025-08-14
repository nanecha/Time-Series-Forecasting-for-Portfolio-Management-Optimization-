#main function
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.strategy_back_testing import load_data, simulate_portfolio, calculate_metrics, plot_cumulative_returns, summarize_performance


def main():
    """
    Main function to execute Task 5 backtesting.
    """
    # Parameters
    tsla_path = 'F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/TSLA_data.csv'
    bnd_path = 'F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/BND_data.csv'
    spy_path = 'F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/SPY_data.csv'
    portfolio_summary_path = 'F:/Time-Series-Forecasting-for-Portfolio-Management-Optimization-/data/dataf/output/portfolio_summary.txt'
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