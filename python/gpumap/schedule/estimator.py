from collections import deque
from dataclasses import dataclass
from statistics import NormalDist
from typing import Deque, Optional, Tuple
import os
import sqlite3
import time as _time


class Estimator:
    """
    Estimates operations that can be performed within a time budget.
    """

    def add_sample(self, ops: int, elapsed_sec: float):
        raise NotImplementedError()

    def estimate_ops(self, time_budget_sec: float, confidence: float = 0.95) -> int:
        raise NotImplementedError()

    def predict_time(self, ops: int, quantile: float = 0.95) -> float:
        raise NotImplementedError()


@dataclass
class FitResult:
    slope_a: float  # sec per op
    intercept_b: float  # fixed overhead
    sigma: float  # residual std
    n_samples: int


@dataclass
class RateFitResult:
    mean_rate: float  # sec per op
    sigma_rate: float  # residual std in sec per op
    n_samples: int


class MeanRateEstimator(Estimator):
    """
    Model:
        t ≈ (mean_rate ± noise_rate) * ops

    Uses a rolling window of per-op time rates (elapsed_sec / ops) and returns
    a conservative ops estimate with a normal-quantile safety margin.
    """

    def __init__(self, window_size: int = 20, min_samples: int = 3):
        self.window_size = window_size
        self.min_samples = min_samples
        self._samples: Deque[Tuple[int, float]] = deque(maxlen=window_size)
        self._fit: Optional[RateFitResult] = None

    def add_sample(self, ops: int, elapsed_sec: float):
        if ops <= 0 or elapsed_sec < 0:
            return
        self._samples.append((ops, elapsed_sec))
        self._fit = None

    def fit(self) -> RateFitResult:
        rates = [elapsed_sec / ops for ops, elapsed_sec in self._samples if ops > 0]
        n = len(rates)
        if n < self.min_samples:
            raise ValueError("Not enough samples.")

        mean_rate = sum(rates) / n
        if mean_rate <= 0:
            raise ValueError("Non-positive mean rate.")

        if n >= 2:
            m = mean_rate
            var = sum((r - m) ** 2 for r in rates) / (n - 1)
            sigma_rate = var**0.5
        else:
            sigma_rate = 0.0

        self._fit = RateFitResult(mean_rate, sigma_rate, n)
        return self._fit

    def fit_result(self) -> RateFitResult:
        return self._fit if self._fit else self.fit()

    def estimate_ops(self, time_budget_sec: float, confidence: float = 0.95) -> int:
        if time_budget_sec <= 0:
            return 0

        fr = self.fit_result()
        z = NormalDist().inv_cdf(confidence)
        safe_rate = fr.mean_rate + z * fr.sigma_rate
        if safe_rate <= 0:
            return 0
        return max(0, int(time_budget_sec / safe_rate))

    def predict_time(self, ops: int, quantile: float = 0.95) -> float:
        if ops <= 0:
            return 0.0
        fr = self.fit_result()
        z = NormalDist().inv_cdf(quantile)
        safe_rate = fr.mean_rate + z * fr.sigma_rate
        return ops * max(0.0, safe_rate)


class DefaultLinearEstimator(Estimator):
    """
    Model:
        t ≈ a * ops + b + noise

    Recent samples are used for linear regression.
    Returns conservative ops count satisfying:
        P(time <= budget) >= confidence
    """

    def __init__(self, window_size: int = 10, min_samples: int = 3):
        self.window_size = window_size
        self.min_samples = min_samples
        self.cold_start_ops = [1000, 2000, 3000]
        self._samples: Deque[Tuple[int, float]] = deque(maxlen=window_size)
        self._fit: Optional[FitResult] = None

        self.cold_cnt = 0

    def add_sample(self, ops: int, elapsed_sec: float):
        if ops <= 0 or elapsed_sec < 0:
            return
        self._samples.append((ops, elapsed_sec))
        self._fit = None

    def fit(self) -> Optional[FitResult]:
        if len(self._samples) < self.min_samples:
            return None

        xs = [o for o, _ in self._samples]
        ys = [t for _, t in self._samples]

        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        sxx = sum((x - mean_x) ** 2 for x in xs)
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

        if sxx == 0:
            raise ValueError("Degenerate samples.")

        a = sxy / sxx
        b = mean_y - a * mean_x

        residuals = [y - (a * x + b) for x, y in zip(xs, ys)]

        if n >= 2:
            m = sum(residuals) / n
            var = sum((r - m) ** 2 for r in residuals) / (n - 1)
            sigma = var**0.5
        else:
            sigma = 0.0

        self._fit = FitResult(a, b, sigma, n)
        return self._fit

    def fit_result(self) -> Optional[FitResult]:
        return self._fit if self._fit else self.fit()

    def _next_cold_start_ops(self) -> int:
        if self.cold_cnt < len(self.cold_start_ops):
            ops = self.cold_start_ops[self.cold_cnt]
            self.cold_cnt += 1
            return ops
        return self.cold_start_ops[-1]

    def _fallback_ops_from_recent_rates(self, time_budget_sec: float) -> int:
        if time_budget_sec <= 0:
            return 0
        if not self._samples:
            return self._next_cold_start_ops()

        # Tiny runs distort per-op latency badly. When the linear fit becomes
        # unusably pessimistic, ignore very small runs and fall back to the
        # slowest observed throughput among the larger recent samples.
        max_ops = max(ops for ops, _ in self._samples)
        min_focus_ops = max_ops * 0.01
        focus = [
            (ops, elapsed)
            for ops, elapsed in self._samples
            if ops >= min_focus_ops and ops > 0 and elapsed > 0
        ]
        rates = [elapsed / ops for ops, elapsed in focus if ops > 0 and elapsed > 0]
        if not rates:
            return self._next_cold_start_ops()

        safe_rate = max(rates)
        if safe_rate <= 0:
            return self._next_cold_start_ops()

        return max(1, int(time_budget_sec / safe_rate))

    def estimate_ops(self, time_budget_sec: float, confidence: float = 0.95) -> int:
        fr = self.fit_result()

        if fr is None:
            return self._next_cold_start_ops()

        a, b, sigma = fr.slope_a, fr.intercept_b, fr.sigma

        if a <= 0:
            return self._fallback_ops_from_recent_rates(time_budget_sec)

        z = NormalDist().inv_cdf(confidence)

        safe_budget = time_budget_sec - b - z * sigma
        if safe_budget <= 0:
            return self._fallback_ops_from_recent_rates(time_budget_sec)

        ops = safe_budget / a

        return int(ops) if ops > 0 else self._fallback_ops_from_recent_rates(
            time_budget_sec
        )

    def predict_time(self, ops: int, quantile: float = 0.95) -> float:
        fr = self.fit_result()
        if fr is None:
            return float("inf")
        z = NormalDist().inv_cdf(quantile)
        return fr.slope_a * ops + fr.intercept_b + z * fr.sigma


class PersistentEstimator(Estimator):
    """
    Wraps any Estimator with SQLite-backed persistence.

    Past samples survive process restarts. On init, up to `load_limit`
    most-recent rows are loaded into the base estimator so it can start
    making predictions immediately without a cold-start warm-up period.

    Parameters
    ----------
    base:
        The underlying estimator to delegate add_sample / estimate_ops /
        predict_time to.
    db_path:
        Path to the SQLite database file. The directory is created if it
        does not exist.
    kind:
        Logical name for this estimator (e.g. "insertion", "update",
        "embedding"). Used as the table key so multiple estimators can
        share the same DB file.
    load_limit:
        How many historical rows to pre-load into `base` on startup.
    """

    def __init__(
        self,
        base: Estimator,
        db_path: str,
        kind: str,
        load_limit: int = 100,
    ):
        self._base = base
        self._kind = kind
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS samples "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
            " kind TEXT NOT NULL, ops INTEGER NOT NULL, elapsed REAL NOT NULL, "
            " ts REAL NOT NULL)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kind_ts ON samples(kind, ts)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS runs "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
            " kind TEXT NOT NULL, data_name TEXT NOT NULL, "
            " n_features INTEGER NOT NULL, n_inserted_instances INTEGER NOT NULL, "
            " ts REAL NOT NULL)"
        )
        self._conn.commit()
        self._load(load_limit)

    def _load(self, limit: int) -> None:
        rows = self._conn.execute(
            "SELECT ops, elapsed FROM samples WHERE kind=? " "ORDER BY ts DESC LIMIT ?",
            (self._kind, limit),
        ).fetchall()
        for ops, elapsed in reversed(rows):
            self._base.add_sample(int(ops), float(elapsed))

    def add_sample(self, ops: int, elapsed_sec: float) -> None:
        if ops <= 0 or elapsed_sec < 0:
            return
        self._base.add_sample(ops, elapsed_sec)
        self._conn.execute(
            "INSERT INTO samples(kind, ops, elapsed, ts) VALUES(?,?,?,?)",
            (self._kind, ops, elapsed_sec, _time.time()),
        )
        self._conn.commit()

    def estimate_ops(self, time_budget_sec: float, confidence: float = 0.95) -> int:
        return self._base.estimate_ops(time_budget_sec, confidence)

    def predict_time(self, ops: int, quantile: float = 0.95) -> float:
        return self._base.predict_time(ops, quantile)

    def log_run(
        self,
        data_name: str,
        n_features: int,
        n_inserted_instances: int,
    ) -> None:
        self._conn.execute(
            "INSERT INTO runs(kind, data_name, n_features, n_inserted_instances, ts) "
            "VALUES(?,?,?,?,?)",
            (self._kind, data_name, n_features, n_inserted_instances, _time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# -----------------------------
# example usage
# -----------------------------
# if __name__ == "__main__":
#     est = DefaultLinearEstimator()

#     est.extend([
#         (1000, 0.9),
#         (1500, 1.25),
#         (2000, 1.6),
#         (2500, 2.05),
#         (3000, 2.4),
#     ])

#     ops95 = est.estimate_ops(2.0, confidence=0.95)
#     print("ops @95%:", ops95)
#     print("predicted 95% time:",
#           est.predict_time(ops95))
