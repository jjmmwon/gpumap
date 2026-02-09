from collections import deque
from dataclasses import dataclass
from statistics import NormalDist
from typing import Deque, Iterable, Optional, Tuple


class Estimator:
    """
    Estimates operations that can be performed within a time budget.
    """

    def add_sample(self, ops: int, elapsed_sec: float):
        raise NotImplementedError()

    def estimate_ops(self,
                     time_budget_sec: float,
                     confidence: float = 0.95) -> int:
        raise NotImplementedError()

@dataclass
class FitResult:
    slope_a: float      # sec per op
    intercept_b: float  # fixed overhead
    sigma: float        # residual std
    n_samples: int

class DefaultLinearEstimator(Estimator):
    """
    Model:
        t ≈ a * ops + b + noise

    Recent samples are used for linear regression.
    Returns conservative ops count satisfying:
        P(time <= budget) >= confidence
    """

    def __init__(self, window_size: int = 20, min_samples: int = 5):
        self.window_size = window_size
        self.min_samples = min_samples
        self._samples: Deque[Tuple[int, float]] = deque(maxlen=window_size)
        self._fit: Optional[FitResult] = None

    def add_sample(self, ops: int, elapsed_sec: float):
        self._samples.append((ops, elapsed_sec))
        self._fit = None

    # def extend(self, samples: Iterable[Tuple[int, float]]):
    #     for s in samples:
    #         self.add_sample(*s)

    def fit(self) -> FitResult:
        if len(self._samples) < self.min_samples:
            raise ValueError("Not enough samples.")

        xs = [o for o, _ in self._samples]
        ys = [t for _, t in self._samples]

        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        sxx = sum((x - mean_x) ** 2 for x in xs)
        sxy = sum((x - mean_x) * (y - mean_y)
                  for x, y in zip(xs, ys))

        if sxx == 0:
            raise ValueError("Degenerate samples.")

        a = sxy / sxx
        b = mean_y - a * mean_x

        residuals = [y - (a * x + b)
                     for x, y in zip(xs, ys)]

        if n >= 2:
            m = sum(residuals) / n
            var = sum((r - m) ** 2 for r in residuals) / (n - 1)
            sigma = var ** 0.5
        else:
            sigma = 0.0

        self._fit = FitResult(a, b, sigma, n)
        return self._fit

    def fit_result(self) -> FitResult:
        return self._fit if self._fit else self.fit()

    def estimate_ops(self,
                     time_budget_sec: float,
                     confidence: float = 0.95) -> int:
        fr = self.fit_result()
        a, b, sigma = fr.slope_a, fr.intercept_b, fr.sigma

        if a <= 0:
            return 0

        z = NormalDist().inv_cdf(confidence)

        safe_budget = time_budget_sec - b - z * sigma
        if safe_budget <= 0:
            return 0

        ops = safe_budget / a
        return max(0, int(ops))


    def predict_time(self,
                     ops: int,
                     quantile: float = 0.95) -> float:
        fr = self.fit_result()
        z = NormalDist().inv_cdf(quantile)
        return fr.slope_a * ops + fr.intercept_b + z * fr.sigma


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