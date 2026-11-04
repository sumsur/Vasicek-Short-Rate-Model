import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Results from MLE and NLS estimation
data = {
    "Regime": [
        "2019-2022",
        "2022-2024",
        "2024-2026",
        "Full sample"
    ],
    "MLE_kappa": [13.196920, 0.231356, 0.696552, 0.156874],
    "NLS_kappa": [9.028332, 0.090556, 0.303365, 0.053292],
    "MLE_theta": [-0.005569, 0.113120, 0.019216, 0.040334],
    "NLS_theta": [-0.005579, 0.144234, 0.019027, 0.057747]
}

df = pd.DataFrame(data)

# Plot 1: Kappa comparison
x = np.arange(len(df))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - width/2, df["MLE_kappa"], width, label="MLE")
plt.bar(x + width/2, df["NLS_kappa"], width, label="NLS")

plt.xticks(x, df["Regime"])
plt.ylabel("Kappa (per year)")
plt.title("Vasicek Model: MLE vs NLS – Mean Reversion Speed")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("vasicek_mle_nls_kappa.png", dpi=300)
plt.show()


# Plot 2: Theta comparison
plt.figure(figsize=(10, 6))

plt.bar(x - width/2, df["MLE_theta"] * 100, width, label="MLE")
plt.bar(x + width/2, df["NLS_theta"] * 100, width, label="NLS")

plt.xticks(x, df["Regime"])
plt.ylabel("Theta (%)")
plt.title("Vasicek Model: MLE vs NLS – Long-Term Mean Rate")
plt.axhline(0, linewidth=0.8)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("vasicek_mle_nls_theta.png", dpi=300)
plt.show()

print("Comparison charts successfully saved.")
