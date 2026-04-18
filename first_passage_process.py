import numpy as np
import multiprocessing
from typing import Dict
from enum import Enum
import numba


class PassageMethod(Enum):
    """Enumeration for different first-passage calculations."""
    EITHER_SIDE = "either"
    LEFT_SIDE = "left"
    RIGHT_SIDE = "right"


@numba.njit(cache=True)
def _simulate_exit_once(
    x0: float,
    left_side: float,
    right_side: float,
    speed: float,
    lambda_rate: float,
    max_simulation_time: float,
    seed: int,
) -> tuple[float, int]:
    """
    Simulate one telegraph path until the *first* boundary hit.

    Returns
    -------
    (tau, side)
        tau  : first exit time, or NaN if max_simulation_time is exceeded.
        side : -1 if left boundary hit first, +1 if right boundary hit first,
               0 if no hit before max_simulation_time.
    """
    np.random.seed(seed)

    x = x0
    t = 0.0

    # Random initial velocity direction: essential for unbiased MC.
    direction = -1.0 if np.random.random() < 0.5 else 1.0

    while t < max_simulation_time:
        if direction > 0.0:
            time_to_boundary = (right_side - x) / speed
            hit_side = 1
        else:
            time_to_boundary = (x - left_side) / speed
            hit_side = -1

        # Numerical guard
        if time_to_boundary < 0.0:
            time_to_boundary = 0.0

        time_to_flip = np.random.exponential(1.0 / lambda_rate)

        if time_to_boundary <= time_to_flip:
            t += time_to_boundary
            return t, hit_side

        x += direction * speed * time_to_flip
        t += time_to_flip
        direction = -direction

    return np.nan, 0


@numba.njit(parallel=True, cache=True)
def _simulate_many_paths(
    num_paths: int,
    x0: float,
    left_side: float,
    right_side: float,
    speed: float,
    lambda_rate: float,
    max_simulation_time: float,
    seed_offset: int,
) -> np.ndarray:
    """
    Simulate all paths once and compute all relevant first-passage outputs.

    Columns of returned array:
    0: path_id
    1: either_time
    2: left_time   (NaN unless left hit first)
    3: right_time  (NaN unless right hit first)
    4: left_hit_indicator
    5: right_hit_indicator
    """
    out = np.empty((num_paths, 6), dtype=np.float64)

    for i in numba.prange(num_paths):
        tau, side = _simulate_exit_once(
            x0,
            left_side,
            right_side,
            speed,
            lambda_rate,
            max_simulation_time,
            seed_offset + i,
        )

        out[i, 0] = i
        out[i, 1] = tau

        if side == -1:
            out[i, 2] = tau
            out[i, 3] = np.nan
            out[i, 4] = 1.0
            out[i, 5] = 0.0
        elif side == 1:
            out[i, 2] = np.nan
            out[i, 3] = tau
            out[i, 4] = 0.0
            out[i, 5] = 1.0
        else:
            out[i, 2] = np.nan
            out[i, 3] = np.nan
            out[i, 4] = 0.0
            out[i, 5] = 0.0

    return out


class TelegraphProcess:
    """
    Monte Carlo simulation for first-passage of the telegraph process.

    Key points:
    - each trajectory is simulated exactly once,
    - the simulation stops at the first boundary hit,
    - from the same path we recover:
        * time to either boundary,
        * time to left boundary first,
        * time to right boundary first,
        * splitting probabilities.
    """

    def __init__(
        self,
        x0: float,
        left_side: float,
        right_side: float,
        speed: float,
        lambda_rate: float,
        num_paths: int,
        passage_method: PassageMethod,
        num_processes: int | None = None,
        max_simulation_time: float = 1000.0,
        seed_offset: int = 0,
    ):
        if left_side >= right_side:
            raise ValueError("left_side must be smaller than right_side")
        if not (left_side < x0 < right_side):
            raise ValueError("x0 must lie strictly inside (left_side, right_side)")
        if speed <= 0.0:
            raise ValueError("speed must be positive")
        if lambda_rate <= 0.0:
            raise ValueError("lambda_rate must be positive")
        if num_paths <= 0:
            raise ValueError("num_paths must be positive")
        if max_simulation_time <= 0.0:
            raise ValueError("max_simulation_time must be positive")

        self.x0 = x0
        self.left_side = left_side
        self.right_side = right_side
        self.speed = speed
        self.lambda_rate = lambda_rate
        self.num_paths = num_paths
        self.passage_method = passage_method
        self.num_processes = num_processes or multiprocessing.cpu_count()
        self.max_simulation_time = max_simulation_time
        self.seed_offset = seed_offset

        # numba thread count can be tuned externally if desired
        try:
            numba.set_num_threads(max(1, min(self.num_processes, multiprocessing.cpu_count())))
        except Exception:
            pass

        self._cached_results: Dict[str, np.ndarray] | None = None
        self.first_passage_times = np.empty(0, dtype=np.float64)

    def run_all_passage_simulations(self) -> Dict[str, np.ndarray]:
        """Simulate once and derive all first-passage outputs from the same paths."""
        raw = _simulate_many_paths(
            self.num_paths,
            self.x0,
            self.left_side,
            self.right_side,
            self.speed,
            self.lambda_rate,
            self.max_simulation_time,
            self.seed_offset,
        )

        either = raw[:, 1]
        left = raw[:, 2]
        right = raw[:, 3]
        left_hits = raw[:, 4]
        right_hits = raw[:, 5]

        self._cached_results = {
            "either": either,
            "left_conditional": left,
            "right_conditional": right,
            # aliases that are often useful in MC post-processing
            "left": left,
            "right": right,
            "left_hit_indicator": left_hits,
            "right_hit_indicator": right_hits,
        }
        return self._cached_results

    def run_simulation(self, conditional: bool = False) -> np.ndarray:
        """
        Preserve the original public method but make it use the single-run engine.

        For a genuine first-passage problem on an absorbing interval, the only
        meaningful left/right quantities are the times to hit that boundary first.
        """
        if self._cached_results is None:
            self.run_all_passage_simulations()

        assert self._cached_results is not None

        if self.passage_method == PassageMethod.EITHER_SIDE:
            arr = self._cached_results["either"]
        elif self.passage_method == PassageMethod.LEFT_SIDE:
            arr = self._cached_results["left"]
        else:
            arr = self._cached_results["right"]

        self.first_passage_times = arr[~np.isnan(arr)] if conditional else arr
        return self.first_passage_times

    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Summary statistics from one Monte Carlo run."""
        results = self.run_all_passage_simulations()

        stats: Dict[str, Dict[str, float]] = {}
        for key in ("either", "left_conditional", "right_conditional"):
            times = results[key]
            valid = times[~np.isnan(times)]
            hit_prob = len(valid) / self.num_paths

            if len(valid) == 0:
                stats[key] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "valid_paths": 0,
                    "total_paths": self.num_paths,
                    "hit_probability": hit_prob,
                }
                continue

            stats[key] = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "median": float(np.median(valid)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "q25": float(np.percentile(valid, 25)),
                "q75": float(np.percentile(valid, 75)),
                "valid_paths": int(len(valid)),
                "total_paths": int(self.num_paths),
                "hit_probability": hit_prob,
            }

        # splitting probabilities from the same simulated trajectories
        stats["splitting_probabilities"] = {
            "left_first": float(np.mean(results["left_hit_indicator"])),
            "right_first": float(np.mean(results["right_hit_indicator"])),
        }
        return stats


if __name__ == "__main__":
    import time

    num_paths = 100_000
    x0 = 0.5
    left_side = 0.0
    right_side = 1.0
    speed = 0.1
    lambda_rate = 1.0

    simulator = TelegraphProcess(
        x0=x0,
        left_side=left_side,
        right_side=right_side,
        speed=speed,
        lambda_rate=lambda_rate,
        num_paths=num_paths,
        passage_method=PassageMethod.EITHER_SIDE,
        num_processes=multiprocessing.cpu_count(),
        max_simulation_time=1_000.0,
        seed_offset=12345,
    )

    t0 = time.time()
    stats = simulator.get_all_statistics()
    elapsed = time.time() - t0

    print(f"Completed {num_paths:,} paths in {elapsed:.3f} s")
    print(f"Paths/sec: {num_paths / elapsed:,.0f}")
    print(stats)
