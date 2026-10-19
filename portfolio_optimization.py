import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

price_path = "data/raw/stock_monthly_prices.csv"
returns_path = "data/processed/stock_monthly_returns.csv"

if os.path.exists(returns_path):
    print("-> Found stock_monthly_returns.csv.")
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
elif os.path.exists(price_path):
    print("-> Calculating monthly returns from stock_monthly_prices.csv...")
    prices_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
    returns_df = prices_df.pct_change().dropna()
    os.makedirs("data/processed", exist_ok=True)
    returns_df.to_csv(returns_path)
else:
    raise FileNotFoundError("Data file not found! Please run 'python download_data.py' first.")

growth_tickers = ['AAPL', 'MSFT', 'TSLA', 'QQQ']
market_ticker = 'SPY'

growth_tickers = [t for t in growth_tickers if t in returns_df.columns]
growth_returns = returns_df[growth_tickers]
market_returns = returns_df[market_ticker] if market_ticker in returns_df.columns else growth_returns.mean(axis=1)

factor_summary = []
residual_variances = {}

X = sm.add_constant(market_returns)

for stock in growth_tickers:
    y = growth_returns[stock]
    model = sm.OLS(y, X).fit()
    alpha = model.params['const']
    beta = model.params[market_ticker] if market_ticker in model.params else model.params.iloc[1]
    res_var = model.resid.var() * 12
    
    factor_summary.append({
        'Stock': stock,
        'Alpha (Month)': alpha,
        'Alpha (Annualized)': alpha * 12,
        'Beta_Market': beta,
        'Residual_Var_Annual': res_var
    })
    residual_variances[stock] = res_var

factor_df = pd.DataFrame(factor_summary).set_index('Stock')

print("\n" + "="*70)
print("1. GROWTH STOCKS RISK & ALPHA SUMMARY TABLE")
print("="*70)
print(factor_df.round(4))

cov_sample = growth_returns.cov() * 12

market_var_annual = market_returns.var() * 12
betas = factor_df['Beta_Market'].values
n_assets = len(growth_tickers)

cov_sim_mat = np.zeros((n_assets, n_assets))
for i in range(n_assets):
    for j in range(n_assets):
        if i == j:
            cov_sim_mat[i, j] = (betas[i] ** 2) * market_var_annual + factor_df.iloc[i]['Residual_Var_Annual']
        else:
            cov_sim_mat[i, j] = betas[i] * betas[j] * market_var_annual

cov_sim = pd.DataFrame(cov_sim_mat, index=growth_tickers, columns=growth_tickers)

print("\n" + "="*70)
print("2. SAMPLE COVARIANCE MATRIX (%)")
print("="*70)
print((cov_sample * 100).round(2))

print("\n" + "="*70)
print("3. SINGLE INDEX MODEL (SIM) COVARIANCE MATRIX (%)")
print("="*70)
print((cov_sim * 100).round(2))

rf_rate = 0.03
exp_returns = growth_returns.mean() * 12

def optimize_portfolio(cov_matrix, exp_returns, rf):
    num_assets = len(exp_returns)
    
    def objective(weights):
        p_ret = np.dot(weights, exp_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (p_ret - rf) / p_vol
        return -sharpe

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_weights = num_assets * [1.0 / num_assets]
    
    res = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    opt_weights = res.x
    p_ret = np.dot(opt_weights, exp_returns)
    p_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
    p_sharpe = (p_ret - rf) / p_vol
    
    return opt_weights, p_ret, p_vol, p_sharpe

w_sample, ret_s, vol_s, sharpe_s = optimize_portfolio(cov_sample, exp_returns, rf_rate)
w_sim, ret_sim, vol_sim, sharpe_sim = optimize_portfolio(cov_sim, exp_returns, rf_rate)

summary_weights = pd.DataFrame({
    'Sample Covariance (%)': np.round(w_sample * 100, 2),
    'Single Index Model (%)': np.round(w_sim * 100, 2)
}, index=growth_tickers)

print("\n" + "="*70)
print("4. OPTIMAL PORTFOLIO WEIGHTS FOR GROWTH STOCKS")
print("="*70)
print(summary_weights)

print("\n" + "="*70)
print("5. GROWTH PORTFOLIO PERFORMANCE COMPARISON")
print("="*70)
print(f"[-] Sample Covariance Method:")
print(f"    + Expected Return:          {ret_s*100:.2f}%/year")
print(f"    + Standard Deviation (Risk): {vol_s*100:.2f}%/year")
print(f"    + Sharpe Ratio:             {sharpe_s:.2f}")
print(f"[-] Single Index Model (SIM):")
print(f"    + Expected Return:          {ret_sim*100:.2f}%/year")
print(f"    + Standard Deviation (Risk): {vol_sim*100:.2f}%/year")
print(f"    + Sharpe Ratio:             {sharpe_sim:.2f}")
print("="*70)

