import pandas as pd

bonds_data = {
    'Bond': ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'],
    'Maturity_Years': [1, 2, 3, 5, 8],
    'Annual_Coupon_Rate': [0.00, 0.00, 0.04, 0.035, 0.045],
    'Price': [958.77, 922.78, 1002.49, 986.07, 1040.71],
    'Face_Value': [1000, 1000, 1000, 1000, 1000]
}

bonds_df = pd.DataFrame(bonds_data)

print("--- CASE BONDS INFORMATION ---")
print(bonds_df)

bonds_df.to_csv("data/raw/case_bonds.csv", index=False)
print("\nCase bonds data successfully saved to: data/raw/case_bonds.csv")

