import pandas as pd

# 1. Load raw ECB data
df = pd.read_csv("data.csv")

# 2. Check the dataset
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# 3. Select date and interest rate columns
df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()

# 4. Rename columns
df.columns = ["date", "rate_pct"]

# 5. Convert data types
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["rate_pct"] = pd.to_numeric(df["rate_pct"], errors="coerce")

# 6. Remove invalid observations
df = df.dropna(subset=["date", "rate_pct"])

# 7. Sort chronologically and remove duplicate dates
df = (
    df.sort_values("date")
      .drop_duplicates(subset=["date"])
      .reset_index(drop=True)
)

# 8. Convert percentage to decimal
df["rate"] = df["rate_pct"] / 100

# 9. Save cleaned data
df.to_csv("data_clean.csv", index=False)

# 10. Print summary
print("\nFirst five observations:")
print(df.head())

print("\nLast five observations:")
print(df.tail())

print("\nMissing values:")
print(df.isna().sum())

print("\nNumber of observations:", len(df))
print("Date range:", df["date"].min(), "to", df["date"].max())
print("Mean rate (%):", df["rate_pct"].mean())
print("Minimum rate (%):", df["rate_pct"].min())
print("Maximum rate (%):", df["rate_pct"].max())

print("\nCleaned data saved to estr_clean.csv")
