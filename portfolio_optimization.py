import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

# ==============================================================================
# 1. ĐỌC DỮ LIỆU GIÁ VÀ TÍNH TỶ SUẤT SINH LỜI (RETURNS)
# ==============================================================================
price_path = "data/raw/stock_monthly_prices.csv"
returns_path = "data/processed/stock_monthly_returns.csv"

# Kiểm tra nếu chưa có file returns thì tạo từ file prices thô
if os.path.exists(returns_path):
    print("-> Đã tìm thấy file stock_monthly_returns.csv do Linh chuẩn bị.")
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
elif os.path.exists(price_path):
    print("-> Đang tính toán tỷ suất sinh lời hàng tháng từ stock_monthly_prices.csv...")
    prices_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
    returns_df = prices_df.pct_change().dropna()
    os.makedirs("data/processed", exist_ok=True)
    returns_df.to_csv(returns_path)
else:
    raise FileNotFoundError("Không tìm thấy file dữ liệu! Hãy chạy 'python download_data.py' trước.")

# Danh sách cổ phiếu tăng trưởng (Growth Stocks) và chỉ số thị trường (SPY)
growth_tickers = ['AAPL', 'MSFT', 'TSLA', 'QQQ']
market_ticker = 'SPY'

# Lọc các cột có sẵn trong dữ liệu
growth_tickers = [t for t in growth_tickers if t in returns_df.columns]
growth_returns = returns_df[growth_tickers]
market_returns = returns_df[market_ticker] if market_ticker in returns_df.columns else growth_returns.mean(axis=1)

# ==============================================================================
# 2. TRÍCH XUẤT THÔNG SỐ ALPHA VÀ BETA (SINGLE INDEX REGRESSION)
# ==============================================================================
factor_summary = []
residual_variances = {}

X = sm.add_constant(market_returns)

for stock in growth_tickers:
    y = growth_returns[stock]
    model = sm.OLS(y, X).fit()
    alpha = model.params['const']
    beta = model.params[market_ticker] if market_ticker in model.params else model.params.iloc[1]
    res_var = model.resid.var() * 12 # Quy đổi biến động phi hệ thống theo năm
    
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
print("1. BẢNG TRÍCH XUẤT THÔNG SỐ RỦI RO & ALPHA CỦA CÁC CỔ PHIẾU TĂNG TRƯỞNG")
print("="*70)
print(factor_df.round(4))

# ==============================================================================
# 3. TÍNH MA TRẬN HIỆP PHƯƠNG SAI (SAMPLE COVARIANCE VS SINGLE INDEX MODEL)
# ==============================================================================
# A. Ma trận Hiệp phương sai Mẫu (Sample Covariance Matrix - Annualized)
cov_sample = growth_returns.cov() * 12

# B. Ma trận Hiệp phương sai Mô hình Chỉ số Đơn (Single Index Model - SIM)
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
print("2. MA TRẬN HIỆP PHƯƠNG SAI MẪU (SAMPLE COVARIANCE MATRIX - %)")
print("="*70)
print((cov_sample * 100).round(2))

print("\n" + "="*70)
print("3. MA TRẬN HIỆP PHƯƠNG SAI MÔ HÌNH CHỈ SỐ ĐƠN (SIM COVARIANCE MATRIX - %)")
print("="*70)
print((cov_sim * 100).round(2))

# ==============================================================================
# 4. TỐI ƯU HÓA DANH MỤC TĂNG TRƯỞNG (MAXIMIZE SHARPE RATIO)
# ==============================================================================
rf_rate = 0.03 # Lãi suất phi rủi ro 3%/năm
exp_returns = growth_returns.mean() * 12 # TSSL kỳ vọng quy năm

def optimize_portfolio(cov_matrix, exp_returns, rf):
    num_assets = len(exp_returns)
    
    def objective(weights):
        p_ret = np.dot(weights, exp_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (p_ret - rf) / p_vol
        return -sharpe # Tối đa hóa Sharpe = Tối thiểu hóa (-Sharpe)

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

# ==============================================================================
# 5. KẾT QUẢ TỔNG HỢP VÀ QUYẾT ĐỊNH PHÂN BỔ DÀNH CHO BÁO CÁO CỦA QUỲNH
# ==============================================================================
summary_weights = pd.DataFrame({
    'Sample Covariance (%)': np.round(w_sample * 100, 2),
    'Single Index Model (%)': np.round(w_sim * 100, 2)
}, index=growth_tickers)

print("\n" + "="*70)
print("4. BẢNG TỶ TRỌNG PHÂN BỔ TỐI ƯU CHO CÁC CỔ PHIẾU TĂNG TRƯỞNG")
print("="*70)
print(summary_weights)

print("\n" + "="*70)
print("5. SO SÁNH HIỆU NĂNG DANH MỤC TĂNG TRƯỞNG")
print("="*70)
print(f"[-] Phương pháp Hiệp phương sai Mẫu:")
print(f"    + Tỷ suất sinh lời kỳ vọng: {ret_s*100:.2f}%/năm")
print(f"    + Độ lệch chuẩn (Rủi ro):   {vol_s*100:.2f}%/năm")
print(f"    + Chỉ số Sharpe Ratio:      {sharpe_s:.2f}")
print(f"[-] Mô hình Chỉ số Đơn (SIM):")
print(f"    + Tỷ suất sinh lời kỳ vọng: {ret_sim*100:.2f}%/năm")
print(f"    + Độ lệch chuẩn (Rủi ro):   {vol_sim*100:.2f}%/năm")
print(f"    + Chỉ số Sharpe Ratio:      {sharpe_sim:.2f}")
print("="*70)
