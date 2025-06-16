from math import exp


class LogisticRegressionGD:
    def __init__(self, lr: float, n_epochs: int) -> None:
        self.lr = lr
        self.n_epochs = n_epochs

        self.W: list[float]
        self.b: float

    def dot(self, x: list[float], y: list[float]) -> float:
        return sum([i * j for i, j in zip(x, y)])

    def sigmoid(self, x: float) -> float:
        if x < 0:
            return 1 - 1 / (1 + exp(x))
        return 1 / (1 + exp(-x))

    def _init_weights(self, n_feat: int) -> None:
        self.W = [0.0] * n_feat
        self.b = 0.0

    def _update_weights(self, gw: list[float], gb: float) -> None:
        for i in range(len(self.W)):
            self.W[i] -= self.lr * gw[i]
        self.b -= self.lr * gb

    def train(
        self, X: list[list[float]], y: list[int]
    ) -> dict[str, list[float] | float]:
        n_feat = len(X[0])
        self._init_weights(n_feat)

        for i in range(self.n_epochs):
            self.train_one_epoch(X, y)
            # print(f"Epoch={i},  W={self.W};   b={self.b}")

        return {"weight": self.W, "bias": self.b}

    def train_one_epoch(self, X: list[list[float]], y: list[int]) -> None:
        for i in range(len(X)):
            grad_w, grad_b = self.training_step(X[i], y[i])
            self._update_weights(grad_w, grad_b)

    def training_step(
        self, x: list[float], y: int
    ) -> tuple[list[float], float]:
        grad_w: list[float] = []
        prod = y * (self.dot(x, self.W) + self.b)
        for i in range(len(x)):
            res = (1 - self.sigmoid(prod)) * -y * x[i]
            grad_w.append(res)

        grad_b = (1 - self.sigmoid(prod)) * -y
        return grad_w, grad_b

    def predict(self, xs: list[list[float]]) -> list[int]:
        preds = [self.predict_example(x) for x in xs]
        return preds

    def predict_example(self, x: list[float]) -> int:
        pred = self.dot(self.W, x) + self.b
        return 1 if pred > 0 else -1
