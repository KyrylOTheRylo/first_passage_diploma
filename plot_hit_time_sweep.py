import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep x0 in (left, right) and plot mean hit time for all three scenarios."
    )
    parser.add_argument("--left", type=float, default=0.0, help="Left boundary")
    parser.add_argument("--right", type=float, default=1.0, help="Right boundary")
    parser.add_argument("--speed", type=float, default=0.1, help="Telegraph process speed")
    parser.add_argument("--lambda-rate", type=float, default=1.0, help="Single flip rate lambda")
    parser.add_argument(
        "--lambda-rates",
        type=float,
        nargs="+",
        default=None,
        help="Multiple lambda values (space-separated). If set, overrides --lambda-rate.",
    )
    parser.add_argument("--x-points", type=int, default=30, help="Number of x0 values")
    parser.add_argument("--paths-per-x", type=int, default=10_000, help="MC paths per x0")
    parser.add_argument("--max-time", type=float, default=1000.0, help="Max simulation time")
    parser.add_argument("--num-processes", type=int, default=None, help="Worker processes")
    parser.add_argument("--seed-offset", type=int, default=0, help="Base random seed offset")
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save generated x0/curve data as JSON",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the plot window (useful for batch runs)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.left >= args.right:
        raise ValueError("--left must be smaller than --right")
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive")
    if args.x_points < 2:
        raise ValueError("--x-points must be at least 2")
    if args.paths_per_x <= 0:
        raise ValueError("--paths-per-x must be positive")
    if args.max_time <= 0.0:
        raise ValueError("--max-time must be positive")

    lambda_values = args.lambda_rates if args.lambda_rates is not None else [args.lambda_rate]
    for lam in lambda_values:
        if lam <= 0.0:
            raise ValueError("All lambda values must be positive")

    span = args.right - args.left
    eps = max(1e-12, 1e-9 * span)
    x_values = np.linspace(args.left + eps, args.right - eps, args.x_points)

    all_results = []

    for lambda_idx, lam in enumerate(lambda_values):
        mean_either = np.full(args.x_points, np.nan, dtype=np.float64)
        mean_left = np.full(args.x_points, np.nan, dtype=np.float64)
        mean_right = np.full(args.x_points, np.nan, dtype=np.float64)

        lambda_seed_base = args.seed_offset + lambda_idx * args.x_points * args.paths_per_x
        for i, x0 in enumerate(x_values):
            sim = TelegraphProcess(
                x0=float(x0),
                left_side=args.left,
                right_side=args.right,
                speed=args.speed,
                lambda_rate=lam,
                num_paths=args.paths_per_x,
                passage_method=PassageMethod.EITHER_SIDE,
                num_processes=args.num_processes,
                max_simulation_time=args.max_time,
                seed_offset=lambda_seed_base + i * args.paths_per_x,
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

    if not args.no_show:
        plot_results(all_results, x_values)

    if args.save_json is not None:
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
                "left": args.left,
                "right": args.right,
                "speed": args.speed,
                "lambda_rates": [float(lam) for lam in lambda_values],
                "x_points": args.x_points,
                "paths_per_x": args.paths_per_x,
                "max_time": args.max_time,
                "num_processes": args.num_processes,
                "seed_offset": args.seed_offset,
            },
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved sweep data to {args.save_json}")

    print(f"Generated {len(x_values)} x0 points in ({args.left}, {args.right}).")
    print(
        "Curves per lambda: either, left_conditional, right_conditional "
        f"(lambdas: {', '.join(str(lam) for lam in lambda_values)})"
    )


if __name__ == "__main__":
    main()
