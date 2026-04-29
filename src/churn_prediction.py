import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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

# Target
df["churn"] = df["status"].apply(
    lambda x: 1 if x in ["Cancelled", "Suspended", "Expired"] else 0
)

# Features
numeric_features = [
    "annual_premium",
    "client_age",
    "total_claims",
    "total_damage_amount",
    "total_indemnified",
    "previous_claims"
]

categorical_features = [
    "product",
    "gender",
    "channel",
    "risk_zone",
    "usage"
]

X = df[numeric_features + categorical_features]
y = df["churn"]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:")
print(classification_report(y_test, pred))