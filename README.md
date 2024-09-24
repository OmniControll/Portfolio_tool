# **Stock Portfolio Optimization with Monte Carlo Simulation**

This project focuses on optimizing a stock portfolio using Monte Carlo simulation and efficient frontier plotting. By fetching historical stock data from Yahoo Finance, we calculate monthly returns, volatility, and Sharpe ratios to identify the optimal portfolio allocation. The results are visualized through an interactive efficient frontier chart.

![image](https://github.com/user-attachments/assets/38991b39-4bb2-426e-9897-d268f931301c)


## **Project Overview**

The project consists of several key components:
1. **Data fetching**: Retrieve historical stock prices using the Yahoo Finance API.
2. **Financial calculations**: Calculate monthly returns, expected returns, covariance matrices, and portfolio variance.
3. **Monte Carlo simulation**: Simulate numerous portfolios with different weight allocations to generate returns and risk metrics.
4. **Sharpe ratio optimization**: Use the Sharpe ratio to determine the most efficient portfolio, balancing return and volatility.
5. **Visualization**: Plot the efficient frontier showing the relationship between risk and return for different portfolios.

## **Features**

- **Monte Carlo Simulation**: Simulates thousands of portfolio weight combinations and calculates returns, volatility, and Sharpe ratio.
- **Portfolio Optimization**: Finds the optimal portfolio by maximizing the Sharpe ratio.
- **Efficient Frontier Plot**: Visualize the trade-off between risk and return for different portfolios.
- **Downside Deviation Calculation**: Measure risk by focusing on negative returns.
