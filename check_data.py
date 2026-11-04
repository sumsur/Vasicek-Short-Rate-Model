import pandas as pd

# Load cleaned data
df = pd.read_csv("data_clean.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Calculate calendar days between observations
df["days_between"] = df["date"].diff().dt.days

print("Calendar days between observations:")
print(df["days_between"].value_counts().sort_index())

print("\nLargest gaps:")
print(
    df.loc[df["days_between"] > 3,
           ["date", "days_between", "rate_pct"]]
      .to_string(index=False)
)

print("\nDuplicate dates:", df["date"].duplicated().sum())
print("Missing rates:", df["rate"].isna().sum())

print("\nFirst observations:")
print(df[["date", "rate_pct", "rate"]].head())

