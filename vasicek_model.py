import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 1. Load cleaned historical data
df = pd.read_csv("data_clean.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

dates = df["date"]
rates = df["rate"].to_numpy(dtype=float)

# 2. Calculate actual time intervals in years
dt = dates.diff().dt.days.to_numpy()[1:] / 365.0

# Consecutive observations
r_prev = rates[:-1]
r_next = rates[1:]

# 3. Negative log-likelihood
def negative_log_likelihood(params):
    log_kappa, theta, log_sigma = params

    kappa = np.exp(log_kappa)
    sigma = np.exp(log_sigma)

    decay = np.exp(-kappa * dt)

    mean = theta + (r_prev - theta) * decay

    variance = (
        sigma**2
        * (-np.expm1(-2 * kappa * dt))
        / (2 * kappa)
    )

    residual = r_next - mean

    log_likelihood = -0.5 * np.sum(
        np.log(2 * np.pi * variance)
        + residual**2 / variance
    )

    return -log_likelihood


# 4. Initial parameter estimates
theta_initial = np.mean(rates)
sigma_initial = np.std(np.diff(rates)) / np.sqrt(np.mean(dt))

initial_params = [
    np.log(1.0),
    theta_initial,
    np.log(max(sigma_initial, 1e-6))
]

# 5. Estimate parameters using MLE
result = minimize(
    negative_log_likelihood,
    initial_params,
    method="L-BFGS-B",
    bounds=[
        (np.log(0.001), np.log(100.0)),
        (-0.10, 0.10),
        (np.log(1e-6), np.log(2.0))
    ]
)

if not result.success:
    raise RuntimeError(
        f"MLE optimization failed: {result.message}"
    )

kappa = np.exp(result.x[0])
theta = result.x[1]
sigma = np.exp(result.x[2])

# 6. Calculate mean-reversion half-life
half_life_years = np.log(2) / kappa

# 7. Display results
print("\nVASICEK MODEL — MLE ESTIMATION")
print("-" * 40)
print(f"Kappa (per year): {kappa:.6f}")
print(f"Theta (%): {theta * 100:.4f}")
print(f"Sigma (per √year): {sigma:.6f}")
print(f"Sigma (%): {sigma * 100:.4f}")
print(f"Half-life (years): {half_life_years:.4f}")
print(f"Log-likelihood: {-result.fun:.4f}")
print(f"Observations: {len(rates)}")
print(f"Optimizer success: {result.success}")
