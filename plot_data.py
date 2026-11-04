import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("data_clean.csv", parse_dates=["date"])

# Plot historical €STR
plt.figure(figsize=(12, 6))

plt.plot(
    df["date"],
    df["rate_pct"],
    label="Historical €STR",
    linewidth=1.2
)

plt.axhline(
    df["rate_pct"].mean(),
    linestyle="--",
    label=f"Historical mean: {df['rate_pct'].mean():.2f}%"
)

plt.title("Historical €STR — October 2019 to October 2026")
plt.xlabel("Date")
plt.ylabel("Interest Rate (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("historical_estr.png", dpi=300)
plt.show()
