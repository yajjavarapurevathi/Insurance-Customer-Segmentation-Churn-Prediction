import pandas as pd

# Load datasets
claims = pd.read_csv("data/raw/insurance_simple/claims.csv")
contracts = pd.read_csv("data/raw/insurance_simple/contracts.csv")
vehicles = pd.read_csv("data/raw/insurance_simple/vehicles.csv")

# Claims summary by contract
claims_summary = claims.groupby("contract_id").agg(
    total_claims=("claim_id", "count"),
    total_damage_amount=("damage_amount", "sum"),
    total_indemnified=("indemnified_amount", "sum")
).reset_index()

# Merge contracts + vehicles
master = pd.merge(
    contracts,
    vehicles,
    on="contract_id",
    how="left"
)

# Merge claims summary
master = pd.merge(
    master,
    claims_summary,
    on="contract_id",
    how="left"
)

# Fill missing claims with 0
master["total_claims"] = master["total_claims"].fillna(0)
master["total_damage_amount"] = master["total_damage_amount"].fillna(0)
master["total_indemnified"] = master["total_indemnified"].fillna(0)

# Save processed file
master.to_csv("data/processed/insurance_master.csv", index=False)

print("Master dataset created successfully")
print(master.shape)
print(master.head())