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


def _formula_box(ax, text: str):
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )


def plot_results(all_results, x_values, speed, show_theory=True):
    import matplotlib.pyplot as plt

    for lam, curves in all_results:
        fig_time, ax_time = plt.subplots(figsize=(10, 5.5))
        ax_time.scatter(x_values, curves["either_mc"], label="Either MC", s=22, color="#1f77b4")
        ax_time.scatter(x_values, curves["left_mc"], label="Left conditional MC", s=22, color="#d62728")
        ax_time.scatter(x_values, curves["right_mc"], label="Right conditional MC", s=22, color="#2ca02c")
        if show_theory:
            ax_time.plot(x_values, curves["either_theory"], label="T(y) theory", lw=2, color="#1f77b4")
            ax_time.plot(x_values, curves["left_theory"], label="T_L(y) theory", lw=2, color="#d62728")
            ax_time.plot(x_values, curves["right_theory"], label="T_R(y) theory", lw=2, color="#2ca02c")
            _formula_box(
                ax_time,
                "T(y)=L/(2v)+(a/v^2)y(L-y)\n"
                "T_R(y)=W_R(y)/P_R(y)\n"
                "T_L(y)=(T(y)-W_R(y))/P_L(y)\n"
                f"a={lam}, v={speed}",
            )
        ax_time.set_xlabel("Initial point x0")
        ax_time.set_ylabel("Mean first-passage time")
        ax_time.set_title(f"Mean Exit Times vs x0 (a={lam}, v={speed})")
        ax_time.grid(True, alpha=0.3)
        ax_time.legend()
        fig_time.tight_layout()

        fig_prob, ax_prob = plt.subplots(figsize=(10, 5.5))
        ax_prob.scatter(x_values, curves["p_right_mc"], label="Right first MC", s=22, color="#9467bd")
        ax_prob.scatter(x_values, curves["p_left_mc"], label="Left first MC", s=22, color="#ff7f0e")
        if show_theory:
            ax_prob.plot(x_values, curves["p_right_theory"], label="P_R(y) theory", lw=2, color="#9467bd")
            ax_prob.plot(x_values, curves["p_left_theory"], label="P_L(y) theory", lw=2, color="#ff7f0e")
            _formula_box(ax_prob, f"P_R(y)=(v+2ay)/(2(v+aL))\nP_L(y)=1-P_R(y)\na={lam}, v={speed}")
        ax_prob.set_xlabel("Initial point x0")
        ax_prob.set_ylabel("Splitting probability")
        ax_prob.set_ylim(-0.02, 1.02)
        ax_prob.set_title(f"Splitting Probabilities vs x0 (a={lam}, v={speed})")
        ax_prob.grid(True, alpha=0.3)
        ax_prob.legend()
        fig_prob.tight_layout()

    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep x0 in (left, right) and compare first-passage Monte Carlo with telegrapher theory."
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
    parser.add_argument("--max-time", type=float, default=20000.0, help="Max simulation time")
    parser.add_argument("--num-processes", type=int, default=None, help="Worker processes")
    parser.add_argument("--seed-offset", type=int, default=0, help="Base random seed offset")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save data as JSON")
    parser.add_argument("--save-plots", action="store_true", help="Save PNG plots")
    parser.add_argument("--plot-dir", type=Path, default=Path("figures"), help="Directory for plot files")
    parser.add_argument(
        "--show-theory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show theoretical overlays",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows")
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

        curves = {
            "either_mc": mean_either,
            "left_mc": mean_left,
            "right_mc": mean_right,
            "p_left_mc": p_left_mc,
            "p_right_mc": p_right_mc,
            "either_theory": theory_mean_either(x_values, args.left, args.right, args.speed, lam),
            "left_theory": theory_mean_left_conditional(x_values, args.left, args.right, args.speed, lam),
            "right_theory": theory_mean_right_conditional(x_values, args.left, args.right, args.speed, lam),
            "p_left_theory": theory_p_left(x_values, args.left, args.right, args.speed, lam),
            "p_right_theory": theory_p_right(x_values, args.left, args.right, args.speed, lam),
        }
        all_results.append((lam, curves))

    if not args.no_show:
        plot_results(all_results, x_values, args.speed, show_theory=args.show_theory)

    if args.save_plots:
        import matplotlib.pyplot as plt

        args.plot_dir.mkdir(parents=True, exist_ok=True)
        for lam, curves in all_results:
            fig_time, ax_time = plt.subplots(figsize=(10, 5.5))
            ax_time.scatter(x_values, curves["either_mc"], label="Either MC", s=22, color="#1f77b4")
            ax_time.scatter(x_values, curves["left_mc"], label="Left conditional MC", s=22, color="#d62728")
            ax_time.scatter(x_values, curves["right_mc"], label="Right conditional MC", s=22, color="#2ca02c")
            if args.show_theory:
                ax_time.plot(x_values, curves["either_theory"], label="T(y) theory", lw=2, color="#1f77b4")
                ax_time.plot(x_values, curves["left_theory"], label="T_L(y) theory", lw=2, color="#d62728")
                ax_time.plot(x_values, curves["right_theory"], label="T_R(y) theory", lw=2, color="#2ca02c")
            ax_time.grid(True, alpha=0.3)
            ax_time.legend()
            ax_time.set_xlabel("Initial point x0")
            ax_time.set_ylabel("Mean first-passage time")
            ax_time.set_title(f"Mean Exit Times vs x0 (a={lam}, v={args.speed})")
            fig_time.tight_layout()
            fig_time.savefig(args.plot_dir / f"hit_times_lambda_{lam}.png", dpi=160)
            plt.close(fig_time)

            fig_prob, ax_prob = plt.subplots(figsize=(10, 5.5))
            ax_prob.scatter(x_values, curves["p_right_mc"], label="Right first MC", s=22, color="#9467bd")
            ax_prob.scatter(x_values, curves["p_left_mc"], label="Left first MC", s=22, color="#ff7f0e")
            if args.show_theory:
                ax_prob.plot(x_values, curves["p_right_theory"], label="P_R(y) theory", lw=2, color="#9467bd")
                ax_prob.plot(x_values, curves["p_left_theory"], label="P_L(y) theory", lw=2, color="#ff7f0e")
            ax_prob.grid(True, alpha=0.3)
            ax_prob.legend()
            ax_prob.set_xlabel("Initial point x0")
            ax_prob.set_ylabel("Splitting probability")
            ax_prob.set_ylim(-0.02, 1.02)
            ax_prob.set_title(f"Splitting Probabilities vs x0 (a={lam}, v={args.speed})")
            fig_prob.tight_layout()
            fig_prob.savefig(args.plot_dir / f"split_probs_lambda_{lam}.png", dpi=160)
            plt.close(fig_prob)

    if args.save_json is not None:
        payload = {
            "x0": x_values.tolist(),
            "by_lambda": [
                {
                    "lambda_rate": float(lam),
                    "mc": {
                        "either": curves["either_mc"].tolist(),
                        "left_conditional": curves["left_mc"].tolist(),
                        "right_conditional": curves["right_mc"].tolist(),
                        "left_first": curves["p_left_mc"].tolist(),
                        "right_first": curves["p_right_mc"].tolist(),
                    },
                    "theory": {
                        "either": curves["either_theory"].tolist(),
                        "left_conditional": curves["left_theory"].tolist(),
                        "right_conditional": curves["right_theory"].tolist(),
                        "left_first": curves["p_left_theory"].tolist(),
                        "right_first": curves["p_right_theory"].tolist(),
                    },
                }
                for lam, curves in all_results
            ],
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved sweep data to {args.save_json}")


if __name__ == "__main__":
    main()
