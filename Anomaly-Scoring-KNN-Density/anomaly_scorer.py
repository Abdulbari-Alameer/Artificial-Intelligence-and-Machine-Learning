import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


@dataclass
class ThresholdResult:
    threshold: float
    detected_count: int


class AnomalyScorer:
    """
    Production-grade anomaly scorer based on:

        Score = w1 * D + w2 * R

    D = normalized mean k-NN distance
    R = normalized deviation from density threshold

    Final score is guaranteed in [0, 1].
    """

    def __init__(
        self,
        k: int = 10,
        w1: float = 0.6,
        w2: float = 0.4,
        gamma: float = 1.0,
        percentile_clip: Tuple[int, int] = (1, 99),
    ):

        if k < 1:
            raise ValueError("k must be >= 1")

        if not np.isclose(w1 + w2, 1.0):
            raise ValueError("w1 + w2 must equal 1")

        if gamma <= 0:
            raise ValueError("gamma must be > 0")

        self.k = k
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.gamma = float(gamma)
        self.percentile_clip = percentile_clip

        self._fitted = False

    @staticmethod
    def _validate_input(X: np.ndarray) -> np.ndarray:

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("Input data must be 2-dimensional")

        if len(X) < 2:
            raise ValueError("Dataset must contain at least 2 samples")

        if np.isnan(X).any():
            raise ValueError("Input contains NaN values")

        if np.isinf(X).any():
            raise ValueError("Input contains infinite values")

        return X

    def _robust_normalize(self, x: np.ndarray) -> np.ndarray:

        p_low, p_high = np.percentile(x, self.percentile_clip)

        if np.isclose(p_low, p_high):
            return np.zeros_like(x)

        x = np.clip(x, p_low, p_high)

        return (x - p_low) / (p_high - p_low)

    def fit(self, X: np.ndarray):

        X = self._validate_input(X)

        if self.k >= len(X):
            raise ValueError(
                f"k={self.k} must be smaller than number of samples={len(X)}"
            )

        self.X_ = X

        nbrs = NearestNeighbors(
            n_neighbors=self.k + 1,
            algorithm="auto"
        )

        nbrs.fit(X)

        distances, _ = nbrs.kneighbors(X)

        # Exclude self-neighbor
        knn_dist = distances[:, 1:].mean(axis=1)

        # Distance component
        self.D_ = self._robust_normalize(knn_dist)

        # Density estimation
        sigma = np.std(knn_dist)

        if np.isclose(sigma, 0):
            density = np.ones_like(knn_dist)
        else:
            density = np.exp(
                -(knn_dist ** 2) / (2 * sigma ** 2)
            )

        self.tau_ = np.median(density)

        deviation = np.abs(density - self.tau_)

        # Density deviation component
        self.R_ = self._robust_normalize(deviation)

        self._fitted = True

        return self

    def anomaly_score(self) -> np.ndarray:

        self._check_fitted()

        score = (
            self.w1 * self.D_
            + self.w2 * self.R_
        )

        score = np.power(
            np.clip(score, 0.0, 1.0),
            self.gamma
        )

        return score

    def anomaly_score_nonlinear(self) -> np.ndarray:

        self._check_fitted()

        score = 1 - (1 - self.D_) * (1 - self.R_)

        return np.power(
            np.clip(score, 0.0, 1.0),
            self.gamma
        )

    def compute_threshold(
        self,
        scores: np.ndarray,
        percentile: float = 95
    ) -> ThresholdResult:

        if not (0 < percentile < 100):
            raise ValueError("percentile must be in (0,100)")

        threshold = np.percentile(scores, percentile)

        detected = int((scores >= threshold).sum())

        return ThresholdResult(
            threshold=float(threshold),
            detected_count=detected
        )

    def plot_scores(
        self,
        scores: np.ndarray,
        threshold: float,
        save_dir: str = "outputs"
    ):

        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(scores, bins=40)
        axes[0].axvline(threshold)
        axes[0].set_title("Anomaly Score Distribution")

        axes[1].plot(np.sort(scores))
        axes[1].axhline(threshold)
        axes[1].set_title("Sorted Scores")

        plt.tight_layout()

        path = os.path.join(
            save_dir,
            "score_distribution.png"
        )

        plt.savefig(path, dpi=150)
        plt.close()

    def scatter_plot(
        self,
        scores: np.ndarray,
        threshold: float,
        save_dir: str = "outputs"
    ):

        self._check_fitted()

        if self.X_.shape[1] != 2:
            return

        os.makedirs(save_dir, exist_ok=True)

        mask = scores >= threshold

        plt.figure(figsize=(8, 6))

        plt.scatter(
            self.X_[:, 0],
            self.X_[:, 1],
            c=scores
        )

        plt.scatter(
            self.X_[mask, 0],
            self.X_[mask, 1],
            facecolors="none",
            edgecolors="red",
            s=80
        )

        plt.title("Spatial Anomaly Map")

        path = os.path.join(
            save_dir,
            "scatter_anomaly.png"
        )

        plt.savefig(path, dpi=150)
        plt.close()

    def _check_fitted(self):

        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before scoring"
            )


if __name__ == "__main__":

    np.random.seed(42)

    normal = np.random.randn(200, 2)
    anomalies = np.random.uniform(5, 8, (20, 2))

    X = np.vstack([normal, anomalies])

    model = AnomalyScorer(
        k=10,
        w1=0.6,
        w2=0.4,
        gamma=1.0
    )

    model.fit(X)

    scores = model.anomaly_score()

    threshold_result = model.compute_threshold(
        scores,
        percentile=95
    )

    print(f"Min score: {scores.min():.4f}")
    print(f"Max score: {scores.max():.4f}")
    print(f"Mean score: {scores.mean():.4f}")

    print(
        f"Threshold: {threshold_result.threshold:.4f}"
    )

    print(
        f"Detected anomalies: "
        f"{threshold_result.detected_count}"
    )

    model.plot_scores(
        scores,
        threshold_result.threshold
    )

    model.scatter_plot(
        scores,
        threshold_result.threshold
    )