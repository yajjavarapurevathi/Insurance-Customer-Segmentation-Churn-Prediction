import pandas as pd

# Load data
df = pd.read_csv("data/processed/insurance_master.csv")

# Clean money columns
def clean_money(series):
    return (
        series.astype(str)
        .str.replace("€", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("EUR", "", regex=False)
        .str.replace("euros", "", regex=False)
        .str.strip()
    )

for col in ["annual_premium", "total_damage_amount", "total_indemnified"]:
    df[col] = pd.to_numeric(clean_money(df[col]), errors="coerce")

# Churn flag
df["churn"] = df["status"].apply(
    lambda x: 1 if x in ["Cancelled", "Suspended", "Expired"] else 0
)

# KPI Summary
summary = pd.DataFrame({
    "metric": [
        "Total Customers",
        "Churn Rate %",
        "Average Premium",
        "Total Claims",
        "Total Damage Amount",
        "Total Indemnified"
    ],
    "value": [
        len(df),
        round(df["churn"].mean() * 100, 2),
        round(df["annual_premium"].mean(), 2),
        round(df["total_claims"].sum(), 2),
        round(df["total_damage_amount"].sum(), 2),
        round(df["total_indemnified"].sum(), 2)
    ]
})

summary.to_csv("dashboard/kpi_summary.csv", index=False)

# Product summary
product = df.groupby("product").agg(
    customers=("contract_id", "count"),
    avg_premium=("annual_premium", "mean"),
    churn_rate=("churn", "mean")
).reset_index()

product["churn_rate"] = (product["churn_rate"] * 100).round(2)

product.to_csv("dashboard/product_summary.csv", index=False)

# Risk zone summary
risk = df.groupby("risk_zone").agg(
    customers=("contract_id", "count"),
    avg_premium=("annual_premium", "mean"),
    claims=("total_claims", "sum")
).reset_index()

risk.to_csv("dashboard/risk_zone_summary.csv", index=False)

print("Dashboard files created successfully")