import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import pandas as pd
import plotly.express as px
import os

# This code is modular, with functions handling specific tasks.
# The goal is to handle data fetching, financial calculations, optimization, and visualization.
# The main function will ensure the sequence of operations
# First, let's get the stock data from yfinance and do some basic financial calculations.

def fetch_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return stock_data

# Fetch benchmark data for comparison with the portfolio
def fetch_benchmark_data(ticker='SPY', start_date=None, end_date=None):
    """Fetches benchmark data for comparison."""
    benchmark_data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    benchmark_returns = benchmark_data.pct_change().dropna()
    annualized_return = (1 + benchmark_returns.mean()) ** 12 - 1  # Annualize using compounding
    annualized_volatility = benchmark_returns.std() * np.sqrt(12)  # Annualize the volatility
    return annualized_return, annualized_volatility

def calculate_monthly_returns(stock_data):
    """
    Calculates monthly returns from stock data.

    """
    # Resample the data to monthly frequency ('M' stands for month-end) and calculate the percentage change
    monthly_stock_data = stock_data.resample('M').last()  # Take the last price of each month
    monthly_returns = monthly_stock_data.pct_change().dropna()  # Calculate the percentage change and drop NaN values
    
    return monthly_returns

def calculate_expected_returns(monthly_returns):
    """
    Calculates annualized expected returns using compounded monthly returns.

    """
    # Calculate the mean of monthly returns, then compound to get the annualized return
    expected_returns = (1 + monthly_returns).prod() ** (12 / len(monthly_returns)) - 1  # Compounded annual returns formula
    
    return expected_returns

def calculate_covariance_matrix(monthly_returns):
    covariance_matrix = monthly_returns.cov() * 12  # Annualize the covariance matrix
    return covariance_matrix

def calculate_portfolio_variance(weights, covariance_matrix):
    return np.dot(weights.T, np.dot(covariance_matrix, weights))
    
# Let's start with constructing our Monte Carlo simulation.

def monte_carlo_simulation(expected_returns, covariance_matrix, num_portfolios, annual_risk_free_rate):
    num_assets = len(expected_returns)
    results = np.zeros((num_portfolios, num_assets + 3))  # +3 for return, volatility, and Sharpe ratio
    
    # Convert the annual risk-free rate to a monthly risk-free rate
    monthly_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 12) - 1

    for i in range(num_portfolios):
        # Random weights for our portfolio simulation
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        # Portfolio return (annualized)
        portfolio_return = np.sum(weights * expected_returns)
        
        # Portfolio volatility (annualized)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        
        # Sharpe ratio (annual return minus annual risk-free rate, divided by annual volatility)
        sharpe_ratio = (portfolio_return - annual_risk_free_rate) / portfolio_volatility

        # Store results
        results[i, 0:num_assets] = weights
        results[i, num_assets] = portfolio_return  # Annual return
        results[i, num_assets + 1] = portfolio_volatility  # Annual volatility
        results[i, num_assets + 2] = sharpe_ratio

    # Create a DataFrame for the results
    columns = [f'weight_{asset}' for asset in expected_returns.index] + ['return', 'volatility', 'sharpe_ratio']
    results_df = pd.DataFrame(results, columns=columns)

    return results_df

def optimize_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate):
    num_assets = len(expected_returns)

    # To calculate the optimized (maximized) Sharpe ratio, we need to minimize the negative of it.
    # The function below calculates the variance and return, then outputs the negated Sharpe ratio.

    def objective_function(weights):
        portfolio_variance = calculate_portfolio_variance(weights, covariance_matrix)
        expected_portfolio_return = np.sum(weights * expected_returns)
        neg_sharpe_ratio = - (expected_portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance)
        return neg_sharpe_ratio
    
    # We want only one constraint: the sum of the weights must equal 100%. We use a lambda with x as argument to represent the weights of our assets in the portfolio.
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Bounds for each weight (0, 1)
    bounds = [(0, 1) for _ in range(num_assets)]

    # Starting point
    initial_weights = np.array([1 / num_assets] * num_assets)
    optimized_weights = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimized_weights.x

def plot_efficient_frontier(results_df, benchmark_return, benchmark_volatility, optimized_return, optimized_volatility, plot_path=None):

    # Format the weights and other metrics to reduce the number of decimal places and stack them vertically
    results_df['text'] = results_df.apply(
        lambda row: "<br>".join(
            [f"{col.split('_')[1]}: {row[col]:.2f}" for col in results_df.columns if col.startswith('weight_')]
        ) + f"<br>Annual Return: {row['return']:.4f}<br>Annual Volatility: {row['volatility']:.4f}<br>Sharpe Ratio: {row['sharpe_ratio']:.4f}",
        axis=1
    )

    
    # Plot the data with hover text
    fig = px.scatter(
    results_df,
    x='volatility',
    y='return',
    color='sharpe_ratio',
    title='Efficient Frontier',
    color_continuous_scale='cividis',
    labels={'return': 'Annual Returns', 'volatility': 'Annual Volatility'}
)

# Update hover template to show custom text and remove redundant information
    fig.update_traces(
    hovertemplate='%{hovertext}',
    hovertext=results_df['text']
)
    
    # Highlight benchmark
    fig.add_scatter(
        x=[benchmark_volatility],
        y=[benchmark_return],
        mode='markers',
        marker=dict(size=12, color='green', symbol='x'),
        hovertemplate=f"<b>SPY Benchmark</b><br>Return: {benchmark_return:.2%}<br>Volatility: {benchmark_volatility:.2%}",
        name="SPY Benchmark"
    )
    
    # Highlight optimized portfolio
    fig.add_scatter(
        x=[optimized_volatility],
        y=[optimized_return],
        mode='markers',
        marker=dict(size=12, color='red', symbol='star'),
        hovertemplate=f"<b>Optimized Portfolio</b><br>Return: {optimized_return:.2%}<br>Volatility: {optimized_volatility:.2%}",
        name="Optimized Portfolio"
    )
    
    fig.update_traces(marker=dict(size=7), selector=dict(mode='markers'))
    fig.update_layout(coloraxis_colorbar=dict(title='Sharpe Ratio'))
    
    if plot_path:
        # Save the plot to the specified path
        fig.write_image(plot_path)
    
    fig.show()
    
    return plot_path

# Plot the weight distribution for the optimized portfolio
def plot_weight_distribution(optimized_weights):
    """
    Plots the asset weights of the optimized portfolio.
    """
    weight_df = pd.DataFrame.from_dict(optimized_weights, orient='index', columns=['Weight'])
    fig = px.pie(weight_df, names=weight_df.index, values='Weight', title='Optimized Portfolio Asset Weights')
    fig.show()

# Now, let's run our functions to analyze and optimize our stock portfolio. We'll use a risk-free rate of 2%.
# We also list the simulated portfolios by their Sharpe ratio in descending order and select the top 5.
def analyze_stocks(tickers, start_date, end_date, num_portfolios, risk_free_rate, plot_path=None):
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    monthly_returns = calculate_monthly_returns(stock_data)  # Use monthly returns now
    expected_returns = calculate_expected_returns(monthly_returns)
    covariance_matrix = calculate_covariance_matrix(monthly_returns)
    monte_carlo_results = monte_carlo_simulation(expected_returns, covariance_matrix, num_portfolios, risk_free_rate)
    optimized_weights_np = optimize_sharpe_ratio(expected_returns, covariance_matrix, risk_free_rate)
    optimized_weights_list = optimized_weights_np.tolist()
    optimized_weights = dict(zip(tickers, optimized_weights_list))
    top_portfolios = monte_carlo_results.sort_values(by='sharpe_ratio', ascending=False).head(5)
    
    # Annualize monthly return and volatility
    optimized_return = np.sum(optimized_weights_np * expected_returns)
    optimized_volatility = np.sqrt(np.dot(optimized_weights_np.T, np.dot(covariance_matrix, optimized_weights_np)))
    
    # Fetch benchmark data
    benchmark_return, benchmark_volatility = fetch_benchmark_data(start_date=start_date, end_date=end_date)

    plot_path = plot_efficient_frontier(monte_carlo_results, benchmark_return, benchmark_volatility, optimized_return, optimized_volatility, plot_path)

    # Plot the weight distribution of the optimized portfolio
    plot_weight_distribution(optimized_weights)
    results = {
        "optimized_weights": optimized_weights,
        "top_portfolios": top_portfolios,
        "monte_carlo_results": monte_carlo_results,
        "plot_path": plot_path
    }
    
    return results

# Main function to define parameters
def main():
    portfolio = ['BTC-USD','MSFT', 'COIN']
    start_date = '2022-01-01' # yyyy-dd-mm
    end_date = '2024-12-10' # yyyy-dd-mm
    num_portfolios = 10000
    risk_free_rate = 0.02

    # Define directory and filename
    plot_dir = 'stock_analysis_script'
    plot_filename = "efficient_frontier.png"
    plot_path = os.path.join(plot_dir, plot_filename)

    # Ensure the directory exists
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    results = analyze_stocks(portfolio, start_date, end_date, num_portfolios, risk_free_rate, plot_path)

    return results

# Call main
if __name__ == "__main__":
    main()
