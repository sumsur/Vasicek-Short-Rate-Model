import numpy as np
import pandas as pd
from scipy.optimize import minimize
from itertools import product

# 1. Load cleaned data
df = pd.read_csv("data_clean.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

rates = df["rate"].to_numpy(dtype=float)
dates = df["date"]

# Actual time intervals in years
dt = (
    dates.diff().dt.days.iloc[1:].to_numpy(dtype=float)
    / 365.25
)

r_prev = rates[:-1]
r_next = rates[1:]

# 2. Vasicek negative log-likelihood
def negative_log_likelihood(params):

    log_kappa, theta, log_sigma = params

    kappa = np.exp(log_kappa)
    sigma = np.exp(log_sigma)

    decay = np.exp(-kappa * dt)

    conditional_mean = (
        theta + (r_prev - theta) * decay
    )

    variance = (
        sigma**2
        * (-np.expm1(-2 * kappa * dt))
        / (2 * kappa)
    )

    residuals = r_next - conditional_mean

    log_likelihood = -0.5 * np.sum(
        np.log(2 * np.pi * variance)
        + residuals**2 / variance
    )

    return -log_likelihood


# 3. Different starting values
kappa_starts = [0.01, 0.05, 0.15, 0.50, 1.00]
theta_starts = [0.00, 0.01, 0.02, 0.05, 0.08]

sigma_start = 0.01

results = []

# 4. Run MLE from 25 starting combinations
for kappa_start, theta_start in product(
    kappa_starts, theta_starts
):

    initial_params = [
        np.log(kappa_start),
        theta_start,
        np.log(sigma_start)
    ]

    optimization = minimize(
        negative_log_likelihood,
        initial_params,
        method="L-BFGS-B",
        bounds=[
            (np.log(0.001), np.log(100)),
            (-0.10, 0.10),
            (np.log(1e-6), np.log(2))
        ]
    )

    estimated_kappa = np.exp(optimization.x[0])
    estimated_theta = optimization.x[1]
    estimated_sigma = np.exp(optimization.x[2])

    results.append({
        "Start_kappa": kappa_start,
        "Start_theta": theta_start,
        "Estimated_kappa": estimated_kappa,
        "Estimated_theta": estimated_theta,
        "Estimated_sigma": estimated_sigma,
        "Negative_log_likelihood": optimization.fun,
        "Success": optimization.success
    })


# 5. Display and save results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Negative_log_likelihood"
).reset_index(drop=True)

print("\nVASICEK MLE ROBUSTNESS TEST")
print("--------------------------------")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nBEST ESTIMATE")
print("--------------------------------")

best = results_df.iloc[0]

print(f"Kappa: {best['Estimated_kappa']:.6f}")
print(f"Theta: {best['Estimated_theta'] * 100:.4f}%")
print(f"Sigma: {best['Estimated_sigma']:.6f}")

print(
    "\nSuccessful optimizations:",
    results_df["Success"].sum(),
    "out of",
    len(results_df)
)

results_df.to_csv(
    "vasicek_robustness_results.csv",
    index=False
)

print("\nResults saved to vasicek_robustness_results.csv")
