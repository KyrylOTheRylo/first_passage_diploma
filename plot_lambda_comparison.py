import json
from pathlib import Path

import numpy as np

from first_passage_process import PassageMethod, TelegraphProcess


def plot_results(all_results, x_values):
    import matplotlib.pyplot as plt

    if len(all_results) == 1:
        fig, ax = plt.subplots(figsize=(9, 5))
        lam, curves = all_results[0]
        ax.scatter(x_values, curves["either"], label="Either side", s=28)
        ax.scatter(x_values, curves["left_conditional"], label="Left side first", s=28)
        ax.scatter(x_values, curves["right_conditional"], label="Right side first", s=28)
        ax.set_xlabel("Initial point x0")
        ax.set_ylabel("Mean time to hit")
        ax.set_title(f"First-passage time vs initial point (lambda={lam})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(nrows=len(all_results), ncols=1, figsize=(9, 5 * len(all_results)), sharex=True)
        for i, (lam, curves) in enumerate(all_results):
            ax = axes[i]
            ax.scatter(x_values, curves["either"], label="Either side", s=28)
            ax.scatter(x_values, curves["left_conditional"], label="Left side first", s=28)
            ax.scatter(x_values, curves["right_conditional"], label="Right side first", s=28)
            ax.set_ylabel("Mean time to hit")
            ax.set_title(f"lambda={lam}")
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes[-1].set_xlabel("Initial point x0")
        fig.tight_layout()
        plt.show()


def main():
    # Hardcoded parameters for lambda comparison
    left = 0.0
    right = 1.0
    speed = 0.1
    lambda_values = [1.0, 2.0, 2/3, 4, 0.1]
    x_points = 30
    paths_per_x = 10000
    max_time = 1000.0
    num_processes = None
    seed_offset = 0

    if left >= right:
        raise ValueError("left must be smaller than right")
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    if x_points < 2:
        raise ValueError("x_points must be at least 2")
    if paths_per_x <= 0:
        raise ValueError("paths_per_x must be positive")
    if max_time <= 0.0:
        raise ValueError("max_time must be positive")

    for lam in lambda_values:
        if lam <= 0.0:
            raise ValueError("All lambda values must be positive")

    span = right - left
    eps = max(1e-12, 1e-9 * span)
    x_values = np.linspace(left + eps, right - eps, x_points)

    all_results = []

    for lambda_idx, lam in enumerate(lambda_values):
        mean_either = np.full(x_points, np.nan, dtype=np.float64)
        mean_left = np.full(x_points, np.nan, dtype=np.float64)
        mean_right = np.full(x_points, np.nan, dtype=np.float64)

        lambda_seed_base = seed_offset + lambda_idx * x_points * paths_per_x
        for i, x0 in enumerate(x_values):
            sim = TelegraphProcess(
                x0=float(x0),
                left_side=left,
                right_side=right,
                speed=speed,
                lambda_rate=lam,
                num_paths=paths_per_x,
                passage_method=PassageMethod.EITHER_SIDE,
                num_processes=num_processes,
                max_simulation_time=max_time,
                seed_offset=lambda_seed_base + i * paths_per_x,
            )
            result = sim.run_all_passage_simulations()

            either_valid = result["either"][~np.isnan(result["either"])]
            left_valid = result["left_conditional"][~np.isnan(result["left_conditional"])]
            right_valid = result["right_conditional"][~np.isnan(result["right_conditional"])]

            if either_valid.size > 0:
                mean_either[i] = np.mean(either_valid)
            if left_valid.size > 0:
                mean_left[i] = np.mean(left_valid)
            if right_valid.size > 0:
                mean_right[i] = np.mean(right_valid)

        curves = {
            "either": mean_either,
            "left_conditional": mean_left,
            "right_conditional": mean_right,
        }
        all_results.append((lam, curves))

    # Plot the results
    plot_results(all_results, x_values)

    # Optionally save to JSON
    save_json = Path("lambda_comparison.json")
    payload = {
        "x0": x_values.tolist(),
        "by_lambda": [
            {
                "lambda_rate": float(lam),
                "either": curves["either"].tolist(),
                "left_conditional": curves["left_conditional"].tolist(),
                "right_conditional": curves["right_conditional"].tolist(),
            }
            for lam, curves in all_results
        ],
        "params": {
            "left": left,
            "right": right,
            "speed": speed,
            "lambda_rates": [float(lam) for lam in lambda_values],
            "x_points": x_points,
            "paths_per_x": paths_per_x,
            "max_time": max_time,
            "num_processes": num_processes,
            "seed_offset": seed_offset,
        },
    }
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved comparison data to {save_json}")

    print(f"Generated {len(x_values)} x0 points in ({left}, {right}).")
    print(
        "Curves per lambda: either, left_conditional, right_conditional "
        f"(lambdas: {', '.join(str(lam) for lam in lambda_values)})"
    )


if __name__ == "__main__":
    main()