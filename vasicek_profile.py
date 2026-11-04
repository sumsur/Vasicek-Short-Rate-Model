
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 1. Load data
df = pd.read_csv("data_clean.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

rates = df["rate"].to_numpy(dtype=float)

dt = (
    df["date"].diff().dt.days.iloc[1:].to_numpy(dtype=float)
    / 365.25
)

r_prev = rates[:-1]
r_next = rates[1:]

# 2. Negative log-likelihood for fixed kappa
def objective(params, kappa):

    theta, log_sigma = params
    sigma = np.exp(log_sigma)

    decay = np.exp(-kappa * dt)

    mean = theta + (r_prev - theta) * decay

    variance = (
        sigma**2
        * (-np.expm1(-2 * kappa * dt))
        / (2 * kappa)
    )

    residuals = r_next - mean

    ll = -0.5 * np.sum(
        np.log(2 * np.pi * variance)
        + residuals**2 / variance
    )

    return -ll


# 3. Test different kappa values
kappa_values = np.linspace(0.02, 0.50, 100)

results = []

for kappa in kappa_values:

    initial = [0.04, np.log(0.01)]

    opt = minimize(
        objective,
        initial,
        args=(kappa,),
        method="L-BFGS-B",
        bounds=[
            (-0.10, 0.10),
            (np.log(1e-6), np.log(2))
        ]
    )

    results.append({
        "kappa": kappa,
        "theta": opt.x[0],
        "sigma": np.exp(opt.x[1]),
        "negative_log_likelihood": opt.fun,
        "success": opt.success
    })


# 4. Save results
profile_df = pd.DataFrame(results)

profile_df.to_csv(
    "vasicek_profile_results.csv",
    index=False
)

# 5. Best result
best = profile_df.loc[
    profile_df["negative_log_likelihood"].idxmin()
]

print("\nVASICEK PROFILE LIKELIHOOD")
print("--------------------------------")
print(f"Best kappa: {best['kappa']:.6f}")
print(f"Best theta: {best['theta'] * 100:.4f}%")
print(f"Best sigma: {best['sigma']:.6f}")
print(
    "Successful optimizations:",
    profile_df["success"].sum(),
    "out of",
    len(profile_df)
)

# 6. Plot profile likelihood
import matplotlib.pyplot as plt

min_nll = profile_df["negative_log_likelihood"].min()

profile_df["LR_statistic"] = (
    2 * (profile_df["negative_log_likelihood"] - min_nll)
)

plt.figure(figsize=(10, 6))

plt.plot(
    profile_df["kappa"],
    profile_df["LR_statistic"]
)

plt.axhline(
    3.84,
    linestyle="--",
    label="95% reference level"
)

plt.xlabel("Kappa")
plt.ylabel("Likelihood ratio statistic")
plt.title("Vasicek Profile Likelihood for Kappa")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "vasicek_profile_likelihood.png",
    dpi=300
)

plt.show()

#7. Approximate 95% profile likelihood interval

accepted = profile_df[
    profile_df["LR_statistic"] <= 3.84
]

print("\n95% PROFILE LIKELIHOOD INTERVAL")
print("--------------------------------")

print(
    f"Kappa range: "
    f"{accepted['kappa'].min():.6f} - "
    f"{accepted['kappa'].max():.6f}"
)

print(
    f"Theta range: "
    f"{accepted['theta'].min() * 100:.4f}% - "
    f"{accepted['theta'].max() * 100:.4f}%"
)