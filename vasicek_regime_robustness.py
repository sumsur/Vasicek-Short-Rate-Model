import numpy as np
import pandas as pd
from scipy.optimize import minimize
from itertools import product

# 1. Load data
df = pd.read_csv("data_clean.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2. Define regimes
regimes = {
    "2019-2022": ("2019-10-01", "2021-12-31"),
    "2022-2024": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-10-31")
}

# 3. Negative log-likelihood
def negative_log_likelihood(params, rates, dt):

    log_kappa, theta, log_sigma = params

    kappa = np.exp(log_kappa)
    sigma = np.exp(log_sigma)

    r_prev = rates[:-1]
    r_next = rates[1:]

    decay = np.exp(-kappa * dt)

    mean = theta + (r_prev - theta) * decay

    variance = (
        sigma**2
        * (-np.expm1(-2 * kappa * dt))
        / (2 * kappa)
    )

    residuals = r_next - mean

    log_likelihood = -0.5 * np.sum(
        np.log(2 * np.pi * variance)
        + residuals**2 / variance
    )

    return -log_likelihood


# 4. Starting values
kappa_starts = [0.01, 0.05, 0.15, 0.50, 1.00, 5.00, 15.00, 30.00]
theta_starts = [-0.01, 0.00, 0.02, 0.05, 0.10, 0.15]
sigma_start = 0.01

results = []

# 5. Multi-start estimation for each regime
for regime_name, (start, end) in regimes.items():

    period = df[
        (df["date"] >= start) &
        (df["date"] <= end)
    ].copy()

    period = period.sort_values("date").reset_index(drop=True)

    rates = period["rate"].to_numpy(dtype=float)

    dt = (
        period["date"].diff().dt.days.iloc[1:].to_numpy(dtype=float)
        / 365.25
    )

    best_result = None

    for kappa_start, theta_start in product(
        kappa_starts, theta_starts
    ):

        initial = [
            np.log(kappa_start),
            theta_start,
            np.log(sigma_start)
        ]

        opt = minimize(
            negative_log_likelihood,
            initial,
            args=(rates, dt),
            method="L-BFGS-B",
            bounds=[
                (np.log(0.001), np.log(100)),
                (-0.10, 0.20),
                (np.log(1e-6), np.log(2))
            ]
        )

        if opt.success and (
            best_result is None or opt.fun < best_result.fun
        ):
            best_result = opt

    if best_result is None:
        print(f"Optimization failed: {regime_name}")
        continue

    kappa = np.exp(best_result.x[0])
    theta = best_result.x[1]
    sigma = np.exp(best_result.x[2])

    results.append({
        "Regime": regime_name,
        "Observations": len(rates),
        "Historical_mean": np.mean(rates),
        "Kappa": kappa,
        "Theta": theta,
        "Sigma": sigma,
        "Half_life_years": np.log(2) / kappa,
        "Negative_log_likelihood": best_result.fun
    })


# 6. Results
results_df = pd.DataFrame(results)

print("\nREGIME MULTI-START MLE RESULTS")
print("--------------------------------")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

results_df.to_csv(
    "vasicek_regime_robustness_results.csv",
    index=False
)

print("\nResults saved successfully.")

