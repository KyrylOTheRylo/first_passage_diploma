import argparse
import json
from pathlib import Path

import numpy as np

from first_passage_process import PassageMethod, TelegraphProcess
from theoretical_telegraph import (
    theory_mean_either,
    theory_mean_left_conditional,
    theory_mean_right_conditional,
    theory_p_left,
    theory_p_right,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Compare first-passage curves across lambda values with theory overlays.")
    parser.add_argument("--left", type=float, default=0.0)
    parser.add_argument("--right", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--lambda-rates", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--x-points", type=int, default=30)
    parser.add_argument("--paths-per-x", type=int, default=10_000)
    parser.add_argument("--max-time", type=float, default=1000.0)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--save-json", type=Path, default=Path("lambda_comparison.json"))
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--show-theory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show theoretical overlays",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def run_sweep(args):
    span = args.right - args.left
    eps = max(1e-12, 1e-9 * span)
    x_values = np.linspace(args.left + eps, args.right - eps, args.x_points)
    by_lambda = []

    for lambda_idx, lam in enumerate(args.lambda_rates):
        mean_either = np.full(args.x_points, np.nan, dtype=np.float64)
        mean_left = np.full(args.x_points, np.nan, dtype=np.float64)
        mean_right = np.full(args.x_points, np.nan, dtype=np.float64)
        p_left_mc = np.full(args.x_points, np.nan, dtype=np.float64)
        p_right_mc = np.full(args.x_points, np.nan, dtype=np.float64)

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

            p_right_mc[i] = np.mean(result["right_hit_indicator"])
            p_left_mc[i] = np.mean(result["left_hit_indicator"])

        by_lambda.append(
            {
                "lambda_rate": float(lam),
                "mc": {
                    "either": mean_either,
                    "left_conditional": mean_left,
                    "right_conditional": mean_right,
                    "left_first": p_left_mc,
                    "right_first": p_right_mc,
                },
                "theory": {
                    "either": theory_mean_either(x_values, args.left, args.right, args.speed, lam),
                    "left_conditional": theory_mean_left_conditional(x_values, args.left, args.right, args.speed, lam),
                    "right_conditional": theory_mean_right_conditional(x_values, args.left, args.right, args.speed, lam),
                    "left_first": theory_p_left(x_values, args.left, args.right, args.speed, lam),
                    "right_first": theory_p_right(x_values, args.left, args.right, args.speed, lam),
                },
            }
        )
    return x_values, by_lambda


def plot_results(x_values, by_lambda, speed, show_theory=True):
    import matplotlib.pyplot as plt

    n = len(by_lambda)
    fig_times, axes_times = plt.subplots(n, 1, figsize=(10, 4.5 * n), sharex=True)
    fig_probs, axes_probs = plt.subplots(n, 1, figsize=(10, 4.5 * n), sharex=True)
    if n == 1:
        axes_times = [axes_times]
        axes_probs = [axes_probs]

    for i, row in enumerate(by_lambda):
        lam = row["lambda_rate"]
        mc = row["mc"]
        th = row["theory"]

        ax_t = axes_times[i]
        ax_t.scatter(x_values, mc["either"], s=18, label="Either MC", color="#1f77b4")
        ax_t.scatter(x_values, mc["left_conditional"], s=18, label="Left cond. MC", color="#d62728")
        ax_t.scatter(x_values, mc["right_conditional"], s=18, label="Right cond. MC", color="#2ca02c")
        if show_theory:
            ax_t.plot(x_values, th["either"], lw=2, label="T(y) theory", color="#1f77b4")
            ax_t.plot(x_values, th["left_conditional"], lw=2, label="T_L(y) theory", color="#d62728")
            ax_t.plot(x_values, th["right_conditional"], lw=2, label="T_R(y) theory", color="#2ca02c")
        ax_t.set_title(f"Mean Exit Times (a={lam}, v={speed})")
        ax_t.grid(True, alpha=0.3)
        ax_t.legend()

        ax_p = axes_probs[i]
        ax_p.scatter(x_values, mc["right_first"], s=18, label="Right first MC", color="#9467bd")
        ax_p.scatter(x_values, mc["left_first"], s=18, label="Left first MC", color="#ff7f0e")
        if show_theory:
            ax_p.plot(x_values, th["right_first"], lw=2, label="P_R(y) theory", color="#9467bd")
            ax_p.plot(x_values, th["left_first"], lw=2, label="P_L(y) theory", color="#ff7f0e")
        ax_p.set_title(f"Splitting Probabilities (a={lam}, v={speed})")
        ax_p.set_ylim(-0.02, 1.02)
        ax_p.grid(True, alpha=0.3)
        ax_p.legend()

    axes_times[-1].set_xlabel("Initial point x0")
    axes_probs[-1].set_xlabel("Initial point x0")
    fig_times.tight_layout()
    fig_probs.tight_layout()
    return fig_times, fig_probs


def main():
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
    if any(lam <= 0.0 for lam in args.lambda_rates):
        raise ValueError("All lambda values must be positive")

    x_values, by_lambda = run_sweep(args)
    fig_times, fig_probs = plot_results(x_values, by_lambda, args.speed, show_theory=args.show_theory)

    if args.save_plots:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        fig_times.savefig(args.plot_dir / "lambda_comparison_times.png", dpi=160)
        fig_probs.savefig(args.plot_dir / "lambda_comparison_probabilities.png", dpi=160)

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()

    payload = {
        "x0": x_values.tolist(),
        "by_lambda": [
            {
                "lambda_rate": row["lambda_rate"],
                "mc": {k: v.tolist() for k, v in row["mc"].items()},
                "theory": {k: v.tolist() for k, v in row["theory"].items()},
            }
            for row in by_lambda
        ],
        "params": {
            "left": args.left,
            "right": args.right,
            "speed": args.speed,
            "lambda_rates": [float(lam) for lam in args.lambda_rates],
            "x_points": args.x_points,
            "paths_per_x": args.paths_per_x,
            "max_time": args.max_time,
            "num_processes": args.num_processes,
            "seed_offset": args.seed_offset,
        },
    }
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved comparison data to {args.save_json}")


if __name__ == "__main__":
    main()
