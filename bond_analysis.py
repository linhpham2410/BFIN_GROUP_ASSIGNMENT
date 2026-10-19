import os
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

# ==============================================================================
def load_case_bonds(filepath="case_bonds.csv"):
    if not os.path.exists(filepath):
        print(f"[INFO] NOT FOUND '{filepath}'. LOADING...")
        default_data = pd.DataFrame({
            "Bond": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
            "Maturity": [1, 2, 3, 5, 8],
            "Coupon": [0.00, 0.00, 0.04, 0.035, 0.045],
            "Price": [958.77, 922.78, 1002.49, 986.07, 1040.71],
            "Face_Value": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        })
        default_data.to_csv(filepath, index=False)

    df = pd.read_csv(filepath)

    col_map = {}
    for col in df.columns:
        c_low = col.strip().lower()
        if "bond" in c_low or "name" in c_low or "id" in c_low:
            col_map[col] = "Bond"
        elif "mat" in c_low or "year" in c_low or "term" in c_low:
            col_map[col] = "Maturity"
        elif "coup" in c_low:
            col_map[col] = "Coupon"
        elif "price" in c_low or "quote" in c_low:
            col_map[col] = "Price"
        elif "face" in c_low or "par" in c_low:
            col_map[col] = "Face_Value"

    df = df.rename(columns=col_map)

    if "Face_Value" not in df.columns:
        df["Face_Value"] = 1000.0

    # Chuẩn hóa kiểu dữ liệu
    if df["Price"].dtype == object:
        df["Price"] = df["Price"].astype(str).str.replace(",", "").str.replace("$", "").astype(float)
    if df["Coupon"].dtype == object:
        df["Coupon"] = df["Coupon"].astype(str).str.replace("%", "").astype(float)

    df["Maturity"] = df["Maturity"].astype(str).str.extract(r"(\d+)").astype(int)

    #ROUND NUMBER
    if df["Coupon"].max() > 0.5:
        df["Coupon"] = df["Coupon"] / 100.0

    return df[["Bond", "Maturity", "Coupon", "Price", "Face_Value"]]

# ==============================================================================
def get_bond_cash_flows(coupon_rate, maturity, face_value=1000.0):
    times = np.arange(1, int(maturity) + 1, dtype=float)
    cfs = np.full(int(maturity), coupon_rate * face_value, dtype=float)
    cfs[-1] += face_value
    return times, cfs


def bond_price_from_ytm(ytm, coupon_rate, maturity, face_value=1000.0):
    times, cfs = get_bond_cash_flows(coupon_rate, maturity, face_value)
    discount_factors = (1.0 + ytm) ** (-times)
    return float(np.sum(cfs * discount_factors))


def calculate_ytm(price, coupon_rate, maturity, face_value=1000.0):
    def objective(y):
        return bond_price_from_ytm(y, coupon_rate, maturity, face_value) - price
    return brentq(objective, -0.5, 1.0, xtol=1e-12)


def calculate_bond_metrics(df_bonds):
    results = []
    for _, row in df_bonds.iterrows():
        bond = row["Bond"]
        T = int(row["Maturity"])
        c = float(row["Coupon"])
        P = float(row["Price"])
        F = float(row["Face_Value"])

        ytm = calculate_ytm(P, c, T, F)
        times, cfs = get_bond_cash_flows(c, T, F)
        pv_cfs = cfs / ((1.0 + ytm) ** times)

        # Macaulay Duration: Sum (t * PV(CF_t)) / Price
        mac_duration = np.sum(times * pv_cfs) / P

        # Modified Duration: Macaulay Duration / (1 + YTM)
        mod_duration = mac_duration / (1.0 + ytm)

        # Convexity: Sum [t * (t + 1) * PV(CF_t)] / [Price * (1 + YTM)^2]
        convexity = np.sum(times * (times + 1.0) * pv_cfs) / (P * ((1.0 + ytm) ** 2))

        results.append({
            "Bond": bond,
            "Maturity_Years": T,
            "Annual_Coupon_Pct": round(c * 100, 2),
            "Price_USD": P,
            "YTM_Pct": round(ytm * 100, 4),
            "Macaulay_Duration": round(mac_duration, 4),
            "Modified_Duration": round(mod_duration, 4),
            "Convexity": round(convexity, 4),
            "_ytm_raw": ytm
        })

    return pd.DataFrame(results)

# ==============================================================================
def bootstrap_incomplete_spot_curve(df_bonds):
    prices = dict(zip(df_bonds["Maturity"], df_bonds["Price"]))
    coupons = dict(zip(df_bonds["Maturity"], df_bonds["Coupon"] * df_bonds["Face_Value"]))
    F = 1000.0

    spot_rates = {}
    # Year 1 (Alpha - Zero coupon)
    spot_rates[1] = (F / prices[1]) ** (1.0 / 1.0) - 1.0
    # Year 2 (Beta - Zero coupon)
    spot_rates[2] = (F / prices[2]) ** (1.0 / 2.0) - 1.0
    # Year 3 (Gamma - 4% coupon)
    c3 = coupons[3]
    pv_1_2 = c3 / (1 + spot_rates[1]) + c3 / ((1 + spot_rates[2]) ** 2)
    spot_rates[3] = ((F + c3) / (prices[3] - pv_1_2)) ** (1.0 / 3.0) - 1.0

    # Year 5 (Delta): s4 = 0.5 * (s3 + s5)
    c5 = coupons[5]
    def eq_s5(s5):
        s4 = 0.5 * (spot_rates[3] + s5)
        pv = (c5 / (1 + spot_rates[1]) +
              c5 / ((1 + spot_rates[2]) ** 2) +
              c5 / ((1 + spot_rates[3]) ** 3) +
              c5 / ((1 + s4) ** 4) +
              (F + c5) / ((1 + s5) ** 5))
        return pv - prices[5]

    spot_rates[5] = brentq(eq_s5, 0.0, 0.5, xtol=1e-12)
    spot_rates[4] = 0.5 * (spot_rates[3] + spot_rates[5])

    # Year 8 (Epsilon)
    c8 = coupons[8]
    def eq_s8(s8):
        s6 = spot_rates[5] + (1.0 / 3.0) * (s8 - spot_rates[5])
        s7 = spot_rates[5] + (2.0 / 3.0) * (s8 - spot_rates[5])
        rates_1_to_8 = [spot_rates[1], spot_rates[2], spot_rates[3],
                        spot_rates[4], spot_rates[5], s6, s7, s8]
        pv = sum(c8 / ((1 + rates_1_to_8[t - 1]) ** t) for t in range(1, 8))
        pv += (F + c8) / ((1 + s8) ** 8)
        return pv - prices[8]

    spot_rates[8] = brentq(eq_s8, 0.0, 0.5, xtol=1e-12)
    spot_rates[6] = spot_rates[5] + (1.0 / 3.0) * (spot_rates[8] - spot_rates[5])
    spot_rates[7] = spot_rates[5] + (2.0 / 3.0) * (spot_rates[8] - spot_rates[5])

    df_curve = pd.DataFrame({
        "Year": range(1, 9),
        "Spot_Rate_Pct": [round(spot_rates[t] * 100, 4) for t in range(1, 9)],
        "Method": [
            "Exact (Alpha)", "Exact (Beta)", "Bootstrapped (Gamma)",
            "Linear Interpolated", "Bootstrapped (Delta)",
            "Linear Interpolated", "Linear Interpolated", "Bootstrapped (Epsilon)"
        ]
    })
    return spot_rates, df_curve

# ==============================================================================
def evaluate_rate_scenarios(df_bonds, df_metrics, spot_rates):
    parallel_records = []
    shocks_bps = [-100, +100, +200]  # -1%, +1%, +2%

    for _, row in df_metrics.iterrows():
        bond = row["Bond"]
        T = int(row["Maturity_Years"])
        c = float(row["Annual_Coupon_Pct"]) / 100.0
        P0 = float(row["Price_USD"])
        y0 = float(row["_ytm_raw"])
        D_mod = float(row["Modified_Duration"])
        Conv = float(row["Convexity"])

        for bps in shocks_bps:
            dy = bps / 10000.0
            # Exact repricing
            P_exact = bond_price_from_ytm(y0 + dy, c, T, 1000.0)
            exact_pct_change = (P_exact - P0) / P0 * 100.0

            # Duration Approximation
            dur_approx_pct = (-D_mod * dy) * 100.0

            # Approximate Duration + Convexity
            dur_conv_approx_pct = (-D_mod * dy + 0.5 * Conv * (dy ** 2)) * 100.0

            parallel_records.append({
                "Bond": bond,
                "Shock_bps": f"{bps:+d} bps",
                "Initial_Price": round(P0, 2),
                "Exact_New_Price": round(P_exact, 4),
                "Exact_Pct_Change": round(exact_pct_change, 4),
                "Duration_Only_Pct": round(dur_approx_pct, 4),
                "Dur_Convexity_Pct": round(dur_conv_approx_pct, 4),
                "Approx_Error_bps": round((dur_conv_approx_pct - exact_pct_change) * 100, 4)
            })

    # Curve Shape Change / Non-Parallel Shift
    curve_records = []
    for scenario_name, s1_bps, s8_bps in [
        ("Steepening (-50bps short, +100bps long)", -50, +100),
        ("Flattening (+100bps short, -50bps long)", +100, -50)
    ]:
        for _, row in df_bonds.iterrows():
            bond = row["Bond"]
            T = int(row["Maturity"])
            c = float(row["Coupon"])
            P0 = float(row["Price"])
            times, cfs = get_bond_cash_flows(c, T, 1000.0)

            new_pv = 0.0
            for t, cf in zip(times, cfs):
                t_int = int(t)
                shock_t = (s1_bps + (t_int - 1) / 7.0 * (s8_bps - s1_bps)) / 10000.0
                new_spot_t = spot_rates[t_int] + shock_t
                new_pv += cf / ((1.0 + new_spot_t) ** t_int)

            pct_change = (new_pv - P0) / P0 * 100.0
            curve_records.append({
                "Scenario": scenario_name,
                "Bond": bond,
                "Maturity": T,
                "Initial_Price": round(P0, 2),
                "Repriced_Spot_Price": round(new_pv, 4),
                "Price_Change_Pct": round(pct_change, 4)
            })

    return pd.DataFrame(parallel_records), pd.DataFrame(curve_records)


# ==============================================================================
# 5. LIABILITY MATCHING
# ==============================================================================
def design_and_simulate_liability_portfolio(df_bonds):
    LIABILITY_2024 = 2_400_000.0  # Năm 2
    LIABILITY_2027 = 3_600_000.0  # Năm 5
    TOTAL_FUND = 12_000_000.0

    bonds_dict = df_bonds.set_index("Bond").to_dict(orient="index")

    # 1. Trả chính xác 3,600,000 USD vào ngày 31/12/2027 (Năm 5) bằng trái phiếu Delta
    delta_cf_year5 = 1000.0 * (1.0 + bonds_dict["Delta"]["Coupon"])
    units_delta = LIABILITY_2027 / delta_cf_year5
    cost_delta = units_delta * bonds_dict["Delta"]["Price"]

    # Tiền lãi coupon hàng năm nhận được từ Delta (từ năm 1 đến năm 4)
    annual_delta_coupon = units_delta * (1000.0 * bonds_dict["Delta"]["Coupon"])

    # 2. Trả chính xác 2,400,000 USD vào ngày 31/12/2024 (Năm 2) bằng coupon Delta + đáo hạn Beta
    remaining_2024_obligation = LIABILITY_2024 - annual_delta_coupon
    units_beta = remaining_2024_obligation / 1000.0
    cost_beta = units_beta * bonds_dict["Beta"]["Price"]

    total_bond_cost = cost_beta + cost_delta
    remaining_for_growth = TOTAL_FUND - total_bond_cost

    # Lập bảng phân bổ danh mục trái phiếu
    df_allocation = pd.DataFrame([
        {
            "Bond": "Alpha",
            "Role": "Not Selected (1Y maturity not needed)",
            "Units_Purchased": 0.0,
            "Price_Per_Unit_USD": bonds_dict["Alpha"]["Price"],
            "Total_Cost_USD": 0.0,
            "Weight_in_Bond_Portfolio_Pct": 0.0,
            "Weight_in_Total_12M_Fund_Pct": 0.0
        },
        {
            "Bond": "Beta",
            "Role": "2024 Scholarship Liability Matching (2Y Zero-Coupon)",
            "Units_Purchased": round(units_beta, 6),
            "Price_Per_Unit_USD": bonds_dict["Beta"]["Price"],
            "Total_Cost_USD": round(cost_beta, 2),
            "Weight_in_Bond_Portfolio_Pct": round(cost_beta / total_bond_cost * 100, 2),
            "Weight_in_Total_12M_Fund_Pct": round(cost_beta / TOTAL_FUND * 100, 2)
        },
        {
            "Bond": "Gamma",
            "Role": "Not Selected (3Y maturity does not match liability dates)",
            "Units_Purchased": 0.0,
            "Price_Per_Unit_USD": bonds_dict["Gamma"]["Price"],
            "Total_Cost_USD": 0.0,
            "Weight_in_Bond_Portfolio_Pct": 0.0,
            "Weight_in_Total_12M_Fund_Pct": 0.0
        },
        {
            "Bond": "Delta",
            "Role": "2027 Learning Centre Liability Matching (5Y 3.5% Coupon)",
            "Units_Purchased": round(units_delta, 6),
            "Price_Per_Unit_USD": bonds_dict["Delta"]["Price"],
            "Total_Cost_USD": round(cost_delta, 2),
            "Weight_in_Bond_Portfolio_Pct": round(cost_delta / total_bond_cost * 100, 2),
            "Weight_in_Total_12M_Fund_Pct": round(cost_delta / TOTAL_FUND * 100, 2)
        },
        {
            "Bond": "Epsilon",
            "Role": "Not Selected (8Y maturity exposes 2027 payment to resale price risk)",
            "Units_Purchased": 0.0,
            "Price_Per_Unit_USD": bonds_dict["Epsilon"]["Price"],
            "Total_Cost_USD": 0.0,
            "Weight_in_Bond_Portfolio_Pct": 0.0,
            "Weight_in_Total_12M_Fund_Pct": 0.0
        }
    ])

    # Lập bảng mô phỏng dòng tiền từng năm từ 2023 đến 2027
    years = [1, 2, 3, 4, 5]
    dates = ["31/12/2023", "31/12/2024", "31/12/2025", "31/12/2026", "31/12/2027"]
    obligations = [0.0, LIABILITY_2024, 0.0, 0.0, LIABILITY_2027]

    cf_rows = []
    cumulative_surplus_cash = 0.0

    for y, dt, ob in zip(years, dates, obligations):
        cf_beta = units_beta * 1000.0 if y == 2 else 0.0
        cf_delta_coupon = annual_delta_coupon
        cf_delta_principal = units_delta * 1000.0 if y == 5 else 0.0
        total_inflow = cf_beta + cf_delta_coupon + cf_delta_principal

        net_cash_flow = total_inflow - ob
        cumulative_surplus_cash += net_cash_flow
        shortfall = max(0.0, ob - total_inflow)

        cf_rows.append({
            "Payment_Date": dt,
            "Year": y,
            "Beta_Cash_Flow_USD": round(cf_beta, 2),
            "Delta_Coupon_USD": round(cf_delta_coupon, 2),
            "Delta_Principal_USD": round(cf_delta_principal, 2),
            "Total_Bond_Inflow_USD": round(total_inflow, 2),
            "Required_Commitment_USD": round(ob, 2),
            "Net_Surplus_Cash_USD": round(net_cash_flow, 2),
            "Shortfall_USD": round(shortfall, 2),
            "Cumulative_Surplus_Buffer_USD": round(cumulative_surplus_cash, 2)
        })

    df_cashflow_sim = pd.DataFrame(cf_rows)

    summary_info = {
        "total_bond_cost": total_bond_cost,
        "remaining_for_growth": remaining_for_growth,
        "units_beta": units_beta,
        "units_delta": units_delta,
        "cost_beta": cost_beta,
        "cost_delta": cost_delta
    }

    return df_allocation, df_cashflow_sim, summary_info


# ==============================================================================
# 6. Xuất TOÀN BỘ PHÂN TÍCH & XUẤT BÁO CÁO
# ==============================================================================
def main():
    print("=" * 85)
    print("KAPPA FOUNDATION - BOND VALUATION & FUNDING RISK ANALYSIS (31 DEC 2022)")
    print("=" * 85)

    # Step 1: Analysis
    df_bonds = load_case_bonds("case_bonds.csv")

    # Step 2: Calculate YTM, Duration, Convexity
    df_metrics = calculate_bond_metrics(df_bonds)
    spot_rates, df_spot_curve = bootstrap_incomplete_spot_curve(df_bonds)

    # Ghép Spot Rate tương ứng với kỳ hạn của từng mã trái phiếu vào bảng chính
    df_metrics["Bootstrapped_Spot_Pct"] = df_metrics["Maturity_Years"].map(
        lambda t: round(spot_rates[int(t)] * 100, 4)
    )

    display_cols = [
        "Bond", "Maturity_Years", "Annual_Coupon_Pct", "Price_USD",
        "YTM_Pct", "Bootstrapped_Spot_Pct", "Macaulay_Duration",
        "Modified_Duration", "Convexity"
    ]
    print("\n[1] YTM, SPOT RATE, DURATION & CONVEXITY:")
    print(df_metrics[display_cols].to_string(index=False))

    print("\n[2] BOOTSTRAPPED & INTERPOLATED SPOT CURVE (YEARS 1 TO 8):")
    print(df_spot_curve.to_string(index=False))

    # Bước 3: Kiểm định Duration-Convexity vs Exact Repricing
    df_parallel, df_curve_shift = evaluate_rate_scenarios(df_bonds, df_metrics, spot_rates)
    print("\n[3] DURATION-CONVEXITY APPROXIMATION VS. EXACT REPRICING (PARALLEL SHIFT):")
    print(df_parallel.to_string(index=False))

    print("\n[4] CURVE SHAPE SCENARIOS (STEEPENING & FLATTENING):")
    print(df_curve_shift.to_string(index=False))

    # Bước 4: Thiết kế danh mục & Mô phỏng dòng tiền trả nợ
    df_alloc, df_cf_sim, summary = design_and_simulate_liability_portfolio(df_bonds)

    print("\n[5] BOND PORTFOLIO ALLOCATION & DEDICATED CASH-FLOW MATCHING:")
    print(df_alloc.to_string(index=False))
    print("-" * 85)
    print(f" -> Total Bond Portfolio Cost             : {summary['total_bond_cost']:14,.2f} USD "
          f"({summary['total_bond_cost']/12_000_000*100:.2f}% of total fund)")
    print(f" -> Remaining Capital for Growth Portfolio : {summary['remaining_for_growth']:14,.2f} USD "
          f"({summary['remaining_for_growth']/12_000_000*100:.2f}% of total fund)")
    print("-" * 85)

    print("\n[6] LIABILITY CASH-FLOW MATCHING SIMULATION (2023 - 2027):")
    print(df_cf_sim.to_string(index=False))

    # Kiểm định tự động (Assertions) để chứng minh không bị thiếu hụt
    inflow_2024 = df_cf_sim.loc[df_cf_sim["Year"] == 2, "Total_Bond_Inflow_USD"].values[0]
    inflow_2027 = df_cf_sim.loc[df_cf_sim["Year"] == 5, "Total_Bond_Inflow_USD"].values[0]
    max_shortfall = df_cf_sim["Shortfall_USD"].max()

    assert np.isclose(inflow_2024, 2_400_000.0, atol=1e-2), "Error: 2024 cash flow mismatch!"
    assert np.isclose(inflow_2027, 3_600_000.0, atol=1e-2), "Error: 2027 cash flow mismatch!"
    assert max_shortfall == 0.0, "Error: Cash shortfall detected!"

    print("\n[VERIFIED] ALL CHECKS PASSED:")
    print(" - Dec 31, 2024: Inflow exactly matches 2,400,000.00 USD commitment (Shortfall = 0.00 USD).")
    print(" - Dec 31, 2027: Inflow exactly matches 3,600,000.00 USD commitment (Shortfall = 0.00 USD).")
    print(" - Surplus Delta coupon cash flows of 121,739.13 USD generated in 2023, 2025, and 2026.")

    os.makedirs("outputs", exist_ok=True)
    df_metrics[display_cols].to_csv("outputs/bond_metrics_summary.csv", index=False)
    df_spot_curve.to_csv("outputs/bootstrapped_spot_curve.csv", index=False)
    df_parallel.to_csv("outputs/duration_convexity_parallel_check.csv", index=False)
    df_curve_shift.to_csv("outputs/curve_shape_scenarios.csv", index=False)
    df_alloc.to_csv("outputs/bond_portfolio_allocation.csv", index=False)
    df_cf_sim.to_csv("outputs/liability_cashflow_simulation.csv", index=False)
    print("\n[SAVED] All 6 analysis tables successfully exported to the 'outputs/' directory.")


if __name__ == "__main__":
    main()