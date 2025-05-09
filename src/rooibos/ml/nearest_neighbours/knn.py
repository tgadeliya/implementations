from typing import Union
import math


class KNNClassifier:
    """Vanilla Implementation of K-Nearest Neighbours Classifier in Pure Python."""

    supported_metrics = ["cosine", "euclidean"]

    def __init__(self, n_neighbors, metric="euclidean") -> None:
        assert (
            metric in self.supported_metrics
        ), f"Unsupported metric: {metric}. Supported metrics are: {self.supported_metrics}"
        self.n_neighbors: int = n_neighbors
        self.metric: str = metric

        self.data: list[list[float]] = None
        self.labels: list[float] = None

    def distance(self, x1: list[float], x2: list[float]) -> float:
        if self.metric == "euclidean":
            return math.sqrt(sum([(x - y) ** 2 for x, y in zip(x1, x2)]))
        elif self.metric == "cosine":
            x1_len = math.sqrt(sum([x**2 for x in x1]))
            x1_norm = [x / x1_len for x in x1]
            x2_len = math.sqrt(sum([x**2 for x in x2]))
            x2_norm = [x / x2_len for x in x2]
            return sum([x1i * x2i for x1i, x2i in zip(x1_norm, x2_norm)])

    def train(self, X: list[list[float]], y: list[float]) -> None:
        self.data = X
        self.labels = {i: label for i, label in enumerate(y)}

    def predict(self, examples: list[list[float]]) -> list[Union[int, str]]:
        predictions = []
        for example in examples:
            pred = self.predict_example(example)
            predictions.append(pred)
        return predictions

    def predict_example(self, example: list[float]) -> Union[int, str]:
        distances = []
        for idx, x in enumerate(self.data):
            d = self.distance(example, x)
            distances.append((idx, d))
        # TODO: add more efficient way to select top-k elements https://en.wikipedia.org/wiki/Quickselect
        distances.sort(key=lambda x: x[1])
        neighbors_idx = [idx for idx, _ in distances[: self.n_neighbors]]
        n_labels = [self.labels[idx] for idx in neighbors_idx]
        return self.aggregate_n_neighbours(n_labels)

    def aggregate_n_neighbours(
        self, n_labels: list[Union[float, str]]
    ) -> Union[int, str]:
        return max(set(n_labels), key=n_labels.count)


class KNNRegressor(KNNClassifier):
    supported_aggregations = ["mean", "median"]

    def __init__(self, n_neighbors, metric="minkowski", aggregate="mean"):
        super().__init__(n_neighbors, metric)
        # TODO: Add ValueError for unsupported metrics
        assert (
            aggregate in self.supported_aggregations
        ), f"Unsupported aggregation: {aggregate}. Supported aggregations are: {self.supported_aggregations}"
        self.aggregate: str = aggregate

    def aggregate_n_neighbours(self, n_labels: list[float]) -> float:
        if self.aggregate == "mean":
            return self._mean_agg(n_labels)
        elif self.aggregate == "median":
            return self._median_agg(n_labels)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregate}")

    def _mean_agg(self, l: list[float]) -> float:
        return sum(l) / len(l)

    def _median_agg(self, l: list[float]) -> float:
        ls = sorted(l)
        n = len(ls)
        # if odd, return the middle element.Otherwise, return the average of the two middle elements
        if n % 2:
            return ls[n // 2]
        else:
            return (ls[n // 2 - 1] + ls[n // 2]) / 2
