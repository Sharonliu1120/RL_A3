import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def smooth(values, window=20):
    return pd.Series(values).rolling(window=window, min_periods=1).mean()


def plot_algorithm_interpolated(csv_paths, output_path, title="Learning Curve", num_points=500, smooth_window=20):
    dfs = [pd.read_csv(path).sort_values("env_steps") for path in csv_paths]

    max_steps = min(df["env_steps"].max() for df in dfs)
    grid = np.linspace(0, max_steps, num_points)

    interpolated_returns = []

    plt.figure(figsize=(8, 5))

    for i, df in enumerate(dfs):
        steps = df["env_steps"].values
        returns = df["return"].values

        interp_returns = np.interp(grid, steps, returns)
        interp_smoothed = smooth(interp_returns, window=smooth_window)

        plt.plot(grid, interp_smoothed, alpha=0.4, label=f"seed {i+1}")
        interpolated_returns.append(interp_returns)

    interpolated_returns = np.array(interpolated_returns)

    mean_returns = interpolated_returns.mean(axis=0)
    std_returns = interpolated_returns.std(axis=0)

    mean_smoothed = smooth(mean_returns, window=smooth_window)
    std_smoothed = smooth(std_returns, window=smooth_window)

    plt.plot(grid, mean_smoothed, linewidth=2, label="mean")
    plt.fill_between(
        grid,
        mean_smoothed - std_smoothed,
        mean_smoothed + std_smoothed,
        alpha=0.2
    )

    plt.xlabel("Environment Steps")
    plt.ylabel("Return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")