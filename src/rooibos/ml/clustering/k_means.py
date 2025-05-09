from random import gauss
from math import sqrt
from collections import defaultdict
from typing import Optional


class KMeans:
    """"""

    def __init__(self, n_clusters: int, max_iter: int = 300) -> None:
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.centroids: Optional[list[list[float]]] = None

    def _init_centroids(self, dim: int) -> list[list[float]]:
        centroids: list[list[float]] = []
        # add random_state support
        for _ in range(self.n_clusters):
            c = [gauss(0, 1) for _ in range(dim)]
            centroids.append(c)
        return centroids

    def train(self, X: list[list[float]]) -> list[list[float]]:
        x_dim = len(X[0])
        self.centroids = self._init_centroids(x_dim)

        n_iter = 0
        while n_iter < self.max_iter:
            new_centroids = self.train_one_epoch(X, self.centroids)
            if self.is_stable(new_centroids, self.centroids):
                break
            self.centroids = new_centroids
            n_iter += 1
            print(f"Epoch={n_iter}, centroids={self.centroids}")
        return self.centroids

    def train_one_epoch(
        self, X: list[list[float]], centroids: list[list[float]]
    ) -> list[list[float]]:
        new_centroids: list[list[float]] = [c for c in centroids]
        centroid2example = self.calculate_clusters_for_dataset(X, centroids)
        for c, examples in centroid2example.items():
            new_centroids[c] = self.agg_examples(examples)
        return new_centroids

    def agg_examples(self, examples: list[list[float]]) -> list[float]:
        n_examples = len(examples)
        examples_mean = [sum(xd) / n_examples for xd in zip(*examples)]
        return examples_mean

    def dist(self, c1: list[float], c2: list[float]) -> float:
        return sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(c1, c2)))

    def calculate_clusters_for_dataset(
        self, X: list[list[float]], centroids: list[list[float]]
    ) -> dict[int, list[list[float]]]:
        c2e: dict[int, list[list[float]]] = defaultdict(list)
        for x in X:
            x_dist = self.calculate_distance(x, centroids)
            x_cent = min(range(self.n_clusters), key=lambda i: x_dist[i])
            c2e[x_cent].append(x)
        return c2e

    def calculate_distance(
        self, e: list[float], centroids: list[list[float]]
    ) -> list[float]:
        return [self.dist(e, c) for c in centroids]

    def is_stable(
        self,
        cs1: list[list[float]],
        cs2: list[list[float]],
        tol: Optional[float] = 1e-5,
    ) -> bool:
        def is_sim_cent(c1: list[float], c2: list[float]) -> bool:
            return all(abs(c1d - c2d) < tol for c1d, c2d in zip(c1, c2))  # type: ignore

        return all(is_sim_cent(c1, c2) for c1, c2 in zip(cs1, cs2))

    # def predict(self, X):
    #     preds = []
    #     for x in X:
    #         prob = self.predict_example(x)
    #         # Manage probs -> preds
    #         pred = prob
    #         preds.append(pred)

    #     return preds

    # def predict_example(self, x):
    #     pass
