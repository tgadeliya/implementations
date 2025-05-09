from abc import ABC, abstractmethod
from typing import Any, Optional
from random import choices


class Bagging(ABC):
    """Vanilla Bootstrap aggregation"""

    def __init__(self, models: list[Any]) -> None:
        self.models: list[Any] = models

        for m in self.models:
            assert m.train, f"Method `.train` is not implemented in model {m}"
            assert (
                m.predict_example
            ), f"Method `.predict_example` is not implemented in model {m}"

        self.n_models: int = len(models)

    def train(self, X: list[list[float]], y: list[float]):
        X_split, y_split = self.split_dataset(X, y, self.n_models)
        models_output: list[Any] = []
        for i in range(self.n_models):
            Xs, ys = X_split[i], y_split[i]
            out = self.models[i].train(Xs, ys)
            models_output.append(out)
        return models_output

    def split_dataset(
        self, X: list[list[float]], y: list[float], n_splits: int
    ) -> tuple[list[list[list[float]]], list[list[float]]]:
        X_split: list[list[list[float]]] = []
        y_split: list[list[float]] = []
        for _ in range(n_splits):
            x_sampled, y_sampled = self.sample_data_bootstrap(X, y)
            X_split.append(x_sampled)
            y_split.append(y_sampled)
        return X_split, y_split

    def sample_data_bootstrap(
        self, X: list[list[float]], y: list[float]
    ) -> tuple[list[list[float]], list[float]]:
        indices = list(range(len(X)))
        n_samples = 100
        indices_sampled = choices(indices, k=n_samples)
        X_sampled = [X[i] for i in indices_sampled]
        y_sampled = [y[i] for i in indices_sampled]
        return X_sampled, y_sampled

    def predict(self, X: list[list[float]]) -> list[Optional[float]]:
        preds: list[Optional[float]] = []
        for x in X:
            pred_models = self.predict_example(x)
            preds.append(self.aggregate(pred_models))
        return preds

    def predict_example(self, x: list[float]) -> list[float]:
        return [m.predict_example(x) for m in self.models]

    @abstractmethod
    def aggregate(self, predictions: list[float]) -> Optional[float]:
        pass


class BaggingClassifier(Bagging):
    """Vanilla Bootstrap aggregation for classification problem with majority voting aggregation"""

    def __init__(self, models: list[Any]) -> None:
        super().__init__(models)

    def aggregate(self, predictions: list[float]) -> Optional[float]:
        return max(set(predictions), key=lambda x: predictions.count(x))


class BaggingRegressor(Bagging):
    """Vanilla Bootstrap aggregation for regression problem"""

    supported_aggregations = ["mean", "median"]

    def __init__(self, models: list[Any], aggregation: str = "mean") -> None:
        super().__init__(models)
        self.aggregation = aggregation
        assert aggregation in self.supported_aggregations

    def aggregate(self, predictions: list[float]) -> Optional[float]:
        if self.aggregation == "mean":
            return self._agg_mean(predictions)
        elif self.aggregation == "median":
            return self._agg_median(predictions)
        else:
            return None

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
