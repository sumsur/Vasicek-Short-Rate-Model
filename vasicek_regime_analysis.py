import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 1. Load data
df = pd.read_csv("data_clean.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2. Define historical regimes
regimes = {
    "2019-2022": ("2019-10-01", "2021-12-31"),
    "2022-2024": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-10-31"),
    "Full sample": ("2019-10-01", "2026-10-31")
}

# 3. Vasicek negative log-likelihood
def negative_log_likelihood(params, rates, dt):

    log_kappa, theta, log_sigma = params

    kappa = np.exp(log_kappa)
    sigma = np.exp(log_sigma)

    r_prev = rates[:-1]
    r_next = rates[1:]

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


# 4. Estimate parameters for each regime
results = []

for regime_name, (start_date, end_date) in regimes.items():

    period = df[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ].copy()

    period = period.sort_values("date").reset_index(drop=True)

    rates = period["rate"].to_numpy(dtype=float)

    dt = (
        period["date"].diff().dt.days.iloc[1:].to_numpy(dtype=float)
        / 365.25
    )

    if len(rates) < 100:
        print(f"Not enough observations: {regime_name}")
        continue

    initial_params = [
        np.log(0.15),
        np.mean(rates),
        np.log(0.01)
    ]

    optimization = minimize(
        negative_log_likelihood,
        initial_params,
        args=(rates, dt),
        method="L-BFGS-B",
        bounds=[
            (np.log(0.001), np.log(100)),
            (-0.10, 0.10),
            (np.log(1e-6), np.log(2))
        ]
    )

    kappa = np.exp(optimization.x[0])
    theta = optimization.x[1]
    sigma = np.exp(optimization.x[2])

    half_life = np.log(2) / kappa

    results.append({
        "Regime": regime_name,
        "Observations": len(rates),
        "Historical_mean": np.mean(rates),
        "Kappa": kappa,
        "Theta": theta,
        "Sigma": sigma,
        "Half_life_years": half_life,
        "Success": optimization.success
    })


# 5. Display results
results_df = pd.DataFrame(results)

print("\nVASICEK REGIME ANALYSIS")
print("--------------------------------")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

# 6. Save results
results_df.to_csv(
    "vasicek_regime_results.csv",
    index=False
)

print("\nResults saved to vasicek_regime_results.csv")
