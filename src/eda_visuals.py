import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("data/processed/insurance_master.csv")

# Clean premium column
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

# 1 Product Distribution
plt.figure(figsize=(8,5))
df["product"].value_counts().plot(kind="bar")
plt.title("Insurance Product Distribution")
plt.xlabel("Product")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/product_distribution.png")
plt.close()

# 2 Policy Status
plt.figure(figsize=(8,5))
df["status"].value_counts().plot(kind="bar")
plt.title("Policy Status Count")
plt.xlabel("Status")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/policy_status.png")
plt.close()

# 3 Average Premium
avg_premium = df.groupby("product")["annual_premium"].mean()

plt.figure(figsize=(8,5))
avg_premium.plot(kind="bar")
plt.title("Average Premium by Product")
plt.xlabel("Product")
plt.ylabel("Premium")
plt.tight_layout()
plt.savefig("images/avg_premium.png")
plt.close()

print("Charts created successfully")