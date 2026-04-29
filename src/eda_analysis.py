import pandas as pd

df = pd.read_csv("data/processed/insurance_master.csv")

# Clean annual_premium
df["annual_premium"] = (
    df["annual_premium"]
    .astype(str)
    .str.replace("€", "", regex=False)
    .str.replace("$", "", regex=False)
    .str.replace("EUR", "", regex=False)
    .str.replace("euros", "", regex=False)
    .str.strip()
)

df["annual_premium"] = pd.to_numeric(df["annual_premium"], errors="coerce")

print("Dataset Shape:", df.shape)

print("\nPolicy Status Count:")
print(df["status"].value_counts())

print("\nProduct Distribution:")
print(df["product"].value_counts())

print("\nAverage Premium by Product:")
print(df.groupby("product")["annual_premium"].mean())

print("\nAverage Age by Product:")
print(df.groupby("product")["client_age"].mean())

print("\nMissing Values:")
print(df.isnull().sum())