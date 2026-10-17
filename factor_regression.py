import pandas as pd
import statsmodels.api as sm

returns_df = pd.read_csv(
    "data/processed/stock_monthly_returns.csv", index_col=0, parse_dates=True
)

factors_df = pd.read_csv(
    "data/raw/F-F_Research_Data_5_Factors_2x3.csv", skiprows=3, index_col=0
)

factors_df = factors_df[factors_df.index.astype(str).str.match(r'^\d{6}$')]

factors_df.index = pd.to_datetime(factors_df.index.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)

for col in factors_df.columns:
    factors_df[col] = pd.to_numeric(factors_df[col], errors='coerce')

returns_df.index = pd.to_datetime(returns_df.index) + pd.offsets.MonthEnd(0)

start_date = "2015-01-01"
end_date = "2022-12-31"

returns_subset = returns_df.loc[start_date:end_date]
factors_subset = factors_df.loc[start_date:end_date]

merged_data = pd.merge(
    returns_subset, factors_subset, left_index=True, right_index=True
)

print("Number of observations after merging:", len(merged_data))

y = pd.to_numeric(merged_data["AAPL"], errors='coerce') - pd.to_numeric(merged_data["RF"], errors='coerce')
X = merged_data[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].apply(pd.to_numeric, errors='coerce')
X = sm.add_constant(X)

reg_data = pd.concat([y, X], axis=1).dropna()
y_clean = reg_data.iloc[:, 0]
X_clean = reg_data.iloc[:, 1:]

print("Number of clean observations for regression:", len(y_clean))

if len(y_clean) > 0:
    model = sm.OLS(y_clean, X_clean).fit()
    print("--- REGRESSION RESULTS FOR AAPL ---")
    print(model.summary())
else:
    print("Error: No data available for regression. Please check date alignment.")
    