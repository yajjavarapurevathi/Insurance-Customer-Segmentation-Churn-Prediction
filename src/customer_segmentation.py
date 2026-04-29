import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("data/processed/insurance_master.csv")

def clean_money_column(series):
    return (
        series.astype(str)
        .str.replace("€", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("EUR", "", regex=False)
        .str.replace("euros", "", regex=False)
        .str.strip()
    )
# Clean premium
df["annual_premium"] = (
    df["annual_premium"]
    .astype(str)
    .str.replace("€", "", regex=False)
    .str.replace("$", "", regex=False)
    .str.replace("EUR", "", regex=False)
    .str.replace("euros", "", regex=False)
    .str.strip()
)
money_columns = ["annual_premium", "total_damage_amount", "total_indemnified"]

for col in money_columns:
    df[col] = clean_money_column(df[col])
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Select features
features = [
    "annual_premium",
    "client_age",
    "total_claims",
    "total_damage_amount",
    "total_indemnified"
]

seg = df[features].fillna(0)

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(seg)

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

# PCA for chart
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1], c=df["cluster"])
plt.title("Customer Segmentation Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.savefig("images/customer_clusters.png")
plt.close()

# Save output
df.to_csv("data/processed/insurance_segmented.csv", index=False)

print("Segmentation completed")
print(df["cluster"].value_counts())
print("\nCluster Summary:")
print(
    df.groupby("cluster")[
        ["annual_premium","client_age","total_claims","total_damage_amount","total_indemnified"]
    ].mean()
)