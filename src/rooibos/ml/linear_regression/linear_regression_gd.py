from typing import Union, Optional


class LinearRegressionSGD:
    def __init__(self, eta, n_epochs):
        self.eta = eta
        self.n_epochs = n_epochs
        self.W: Optional[list[float]] = None
        self.b: Optional[float] = None

    @staticmethod
    def dot(x: list[float], y: list[float]) -> float:
        return sum(map(lambda x: x[0] * x[1], zip(x, y)))

    @staticmethod
    def sum(x: list[float], y: list[float]) -> float:
        return [t[0] + t[1] for t in zip(x, y)]

    def init_weights(self, n_features: int) -> None:
        self.W = [0.0] * n_features
        self.b = 0.0

    def train(
        self, X: list[list[float]], y: list[float]
    ) -> dict[str, Union[list[float], float]]:
        n_feat = len(X[0])
        self.init_weights(n_feat)

        for i in range(self.n_epochs):
            self.train_one_epoch(X, y)
        return {"weight": self.W, "bias": self.b}

    def train_one_epoch(self, X: list[list[float]], y: list[float]) -> None:
        for i in range(len(X)):
            grad_w, grad_b = self.training_step(X[i], y[i])
            self._update_weights(grad_w, grad_b)

    def training_step(self, x: list[float], y: float) -> tuple[list[float], float]:
        y_pred = self.dot(x, self.W) + self.b
        error = y_pred - y
        grad_w = [2 * error * xi for xi in x]
        grad_b = 2 * error
        return grad_w, grad_b

    def _update_weights(self, gw: list[float], gb: float) -> None:
        for i in range(len(self.W)):
            self.W[i] -= self.eta * gw[i]
        self.b -= self.eta * gb

    def predict(self, x: list[float]) -> float:
        return self.b + self.dot(self.W, x)
