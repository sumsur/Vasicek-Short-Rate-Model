import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load cleaned data
df = pd.read_csv("data_clean.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

rates = df["rate"].to_numpy(dtype=float)

# 2. Vasicek model parameters (MLE estimates)
kappa = 0.156965
theta = 0.048332
sigma = 0.007767

# 3. Simulation settings
n_simulations = 10000
n_months = 12
dt = 1 / 12

r0 = rates[-1]

np.random.seed(42)

# 4. Monte Carlo simulation
simulations = np.zeros((n_months + 1, n_simulations))
simulations[0, :] = r0

for t in range(1, n_months + 1):

    previous_rates = simulations[t - 1, :]

    conditional_mean = (
        theta
        + (previous_rates - theta) * np.exp(-kappa * dt)
    )

    conditional_std = np.sqrt(
        sigma**2 * (1 - np.exp(-2 * kappa * dt))
        / (2 * kappa)
    )

    random_shocks = np.random.normal(
        0, 1, n_simulations
    )

    simulations[t, :] = (
        conditional_mean
        + conditional_std * random_shocks
    )

# 5. Save simulation results
simulation_df = pd.DataFrame(
    simulations.T,
    columns=[f"Month_{i}" for i in range(n_months + 1)]
)

simulation_df.to_csv(
    "vasicek_simulation.csv",
    index=False
)

# 6. Calculate terminal rate statistics
terminal_rates = simulations[-1, :]

print("VASICEK MONTE CARLO SIMULATION")
print("--------------------------------")
print(f"Initial rate: {r0 * 100:.3f}%")
print(f"Number of simulations: {n_simulations}")
print(f"Simulation horizon: {n_months} months")
print(f"Mean terminal rate: {np.mean(terminal_rates) * 100:.3f}%")
print(f"Median terminal rate: {np.median(terminal_rates) * 100:.3f}%")
print(f"5th percentile: {np.percentile(terminal_rates, 5) * 100:.3f}%")
print(f"95th percentile: {np.percentile(terminal_rates, 95) * 100:.3f}%")
print(f"Probability of negative rate: {np.mean(terminal_rates < 0) * 100:.2f}%")

# 7. Plot simulated paths
plt.figure(figsize=(12, 6))

for i in range(100):
    plt.plot(
        range(n_months + 1),
        simulations[:, i],
        alpha=0.15
    )

mean_path = np.mean(simulations, axis=1)

plt.plot(
    range(n_months + 1),
    mean_path,
    linewidth=2,
    label="Mean simulated path"
)

plt.axhline(
    theta,
    linestyle="--",
    label="Long-term mean (theta)"
)

plt.title("Vasicek Model - Monte Carlo Simulation")
plt.xlabel("Months")
plt.ylabel("Interest rate")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("vasicek_simulation.png", dpi=300)
plt.show()
