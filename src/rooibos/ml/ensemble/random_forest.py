from abc import ABC, abstractmethod
from typing import Optional, Any
from random import choices

from rooibos.ml.tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForest(ABC):
    def __init__(self, n_trees: int, bootstrap_size: int) -> None:
        self.n_models = n_trees
        self.bootstrap_size = bootstrap_size

        self.models: list[DecisionTreeClassifier | DecisionTreeRegressor] = []

    def train(self, X: list[list[float]], y: list[float]) -> list[Any]:
        n_samples = int(len(X) * self.bootstrap_size)
        X_split, y_split = self.split_dataset_for_bootstrap(X, y, n_samples)
        models_output: list[Any] = []
        for i in range(self.n_models):
            Xs, ys = X_split[i], y_split[i]
            out = self.models[i].train(Xs, ys)
            models_output.append(out)
        return models_output

    def sample_data_bootstrap(
        self, X: list[list[float]], y: list[float], n_samples: int
    ) -> tuple[list[list[float]], list[float]]:
        indices = list(range(len(X)))
        indices_sampled = choices(indices, k=n_samples)
        X_sampled = [X[i] for i in indices_sampled]
        y_sampled = [y[i] for i in indices_sampled]
        return X_sampled, y_sampled

    def split_dataset_for_bootstrap(
        self, X: list[list[float]], y: list[float], n_samples: int
    ) -> tuple[list[list[list[float]]], list[list[float]]]:
        X_split: list[list[list[float]]] = []
        y_split: list[list[float]] = []
        for _ in range(self.n_models):
            x_sampled, y_sampled = self.sample_data_bootstrap(X, y, n_samples=n_samples)
            X_split.append(x_sampled)
            y_split.append(y_sampled)
        return X_split, y_split

    def predict(self, X: list[list[float]]) -> list[Optional[float]]:
        return [self.predict_example(x) for x in X]

    def predict_example(self, x: list[float]) -> float:
        return self.aggregate([m.predict_example(x) for m in self.models])

    @abstractmethod
    def aggregate(self, preds: list[float]) -> float:
        pass


class RandomForestClassifier(RandomForest):
    def __init__(self, n_trees: int, bootstrap_size: int) -> None:
        super().__init__(n_trees, bootstrap_size)
        self.models: list[Any] = [DecisionTreeClassifier(use_random_feature_subset=True, random_feature_subset_size="auto") for _ in range(self.n_models)]

    def aggregate(self, preds: list[float]) -> float:
        return max(set(preds), key=preds.count)


class RandomForestRegressor(RandomForest):
    aggregation_methods = ["mean", "median"]

    def __init__(
        self,
        n_trees: int,
        bootstrap_size: int,
        aggregation_method: str = "mean",
        n_features_per_tree: int = 5,
    ) -> None:
        if aggregation_method not in self.aggregation_methods:
            raise ValueError(
                f"Aggregation method {aggregation_method} not supported. Supported methods are: {self.aggregation_methods}"
            )
        self.aggregation_method = aggregation_method
        super().__init__(n_trees, bootstrap_size)
        self.models: list[Any] = [DecisionTreeRegressor(use_random_feature_subset=True, random_feature_subset_size="auto") for _ in range(self.n_models)]

    def aggregate(self, preds: list[float]) -> float:
        if self.aggregation_method == "median":
            return self._agg_median(preds)
        return self._agg_mean(preds)

    def _agg_mean(self, x: list[float]) -> float:
        return sum(x) / len(x)

    def _agg_median(self, l: list[float]) -> float:
        ls = sorted(l)
        n = len(ls)
        # if odd, return the middle element. Otherwise, return the average of the two middle elements
        if n % 2:
            return ls[n // 2]
        else:
            return (ls[n // 2 - 1] + ls[n // 2]) / 2
