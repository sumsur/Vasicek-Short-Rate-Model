import numpy as np
import pandas as pd


# 1. Load cleaned data
df = pd.read_csv("data_clean.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2. Define regimes
regimes = {
    "2019-2022": ("2019-10-01", "2021-12-31"),
    "2022-2024": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-10-31"),
    "Full sample": ("2019-10-01", "2026-10-31")
}

results = []

# 3. OLS estimation for each regime
for regime_name, (start, end) in regimes.items():

    period = df[
        (df["date"] >= start) &
        (df["date"] <= end)
    ].copy()

    period = period.sort_values("date").reset_index(drop=True)

    r_prev = period["rate"].to_numpy(dtype=float)[:-1]
    r_next = period["rate"].to_numpy(dtype=float)[1:]

    dt = (
        period["date"].diff().dt.days.iloc[1:].to_numpy(dtype=float)
        / 365.25
    )

    # Exact discrete Vasicek conditional mean:
    # E[r(t+dt)] = theta*(1-exp(-kappa*dt))
    # + exp(-kappa*dt)*r(t)

    # Estimate continuous-time parameters by nonlinear least squares
    from scipy.optimize import least_squares

    def residuals(params):

        kappa, theta = params

        decay = np.exp(-kappa * dt)

        predicted = (
            theta + (r_prev - theta) * decay
        )

        return r_next - predicted

    optimization = least_squares(
        residuals,
        x0=[0.15, np.mean(r_prev)],
        bounds=([0.001, -0.10], [100, 0.20])
    )

    kappa = optimization.x[0]
    theta = optimization.x[1]

    predicted = (
        theta
        + (r_prev - theta) * np.exp(-kappa * dt)
    )

    errors = r_next - predicted

    sigma = np.sqrt(
        np.mean(
            errors**2
            * (2 * kappa)
            / (-np.expm1(-2 * kappa * dt))
        )
    )

    half_life = np.log(2) / kappa

    results.append({
        "Regime": regime_name,
        "Observations": len(r_prev),
        "Historical_mean": np.mean(period["rate"]),
        "OLS_kappa": kappa,
        "OLS_theta": theta,
        "OLS_sigma": sigma,
        "Half_life_years": half_life,
        "RMSE": np.sqrt(np.mean(errors**2)),
        "Success": optimization.success
    })


# 4. Display results
results_df = pd.DataFrame(results)

print("\nVASICEK OLS / NONLINEAR LEAST SQUARES RESULTS")
print("---------------------------------------------")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# 5. Save results
results_df.to_csv(
    "vasicek_ols_comparison.csv",
    index=False
)

print("\nResults saved to vasicek_ols_comparison.csv")