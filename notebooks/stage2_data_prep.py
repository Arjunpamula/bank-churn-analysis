import pandas as pd

# 1. LOAD --------------------------------------------------------------------
df = pd.read_csv("Bank_Customer_Churn_Prediction_1.csv")

# 2. FIRST LOOK: size, columns, types ---------------------------------------
print("Shape (rows, columns):", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

# 3. QUALITY CHECKS ----------------------------------------------------------
print("\nMissing values per column:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate customer IDs:", df["customer_id"].duplicated().sum())

# Sanity-check value ranges (catch impossible values)
print("\nNumeric summary:")
print(df.describe().round(2))

# Binary columns should only contain 0 and 1
for col in ["credit_card", "active_member", "churn"]:
    assert set(df[col].unique()) <= {0, 1}, f"{col} has unexpected values"

# Categorical columns: list the categories
print("\nCountries:", df["country"].unique())
print("Genders:", df["gender"].unique())

# 4. BASELINE ----------------------------------------------------------------
baseline = df["churn"].mean()
print(f"\nBaseline churn rate: {baseline:.1%}")

# 5. DROP COLUMNS WE DON'T NEED FOR ANALYSIS --------------------------------
# customer_id is just a label: it carries no information about risk.
# Keep a copy of the IDs in case we want to join results back later.
ids = df["customer_id"]
df = df.drop(columns=["customer_id"])

# 6. FEATURE ENGINEERING: CREATE SEGMENTS -----------------------------------
df["age_group"] = pd.cut(
    df["age"],
    bins=[17, 30, 40, 50, 60, 100],
    labels=["18-30", "31-40", "41-50", "51-60", "60+"],
)

df["tenure_band"] = pd.cut(
    df["tenure"],
    bins=[-1, 2, 5, 8, 10],
    labels=["0-2 yrs", "3-5 yrs", "6-8 yrs", "9-10 yrs"],
)

# Balance: separate zero-balance customers, then split the rest into thirds
df["balance_band"] = "Zero"
nonzero = df["balance"] > 0
df.loc[nonzero, "balance_band"] = pd.qcut(
    df.loc[nonzero, "balance"], q=3, labels=["Low", "Medium", "High"]
).astype(str)

df["product_group"] = df["products_number"].map(
    {1: "1 product", 2: "2 products", 3: "3+ products", 4: "3+ products"}
)

# Readable labels for Power BI
df["active_label"] = df["active_member"].map({1: "Active", 0: "Inactive"})
df["credit_card_label"] = df["credit_card"].map({1: "Has card", 0: "No card"})

# 7. CHECK THE NEW COLUMNS ---------------------------------------------------
for col in ["age_group", "tenure_band", "balance_band", "product_group"]:
    print(f"\n{col} counts:")
    print(df[col].value_counts(dropna=False).sort_index())

assert df[["age_group", "tenure_band", "balance_band", "product_group"]].isna().sum().sum() == 0

# 8. SAVE CLEAN DATA ---------------------------------------------------------
df.to_csv("churn_clean.csv", index=False)
print("\nSaved churn_clean.csv with shape", df.shape)
