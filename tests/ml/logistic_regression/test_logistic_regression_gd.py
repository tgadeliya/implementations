from rooibos.ml.logistic_regression.logistic_regression_gd import (
    LogisticRegressionGD,
)


def test_logistic_regression_gd_init():
    LogisticRegressionGD(lr=0.01, n_epochs=10)


def test_logistic_regression_gd_train_small():
    X = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    y = [1, 1, -1]
    model = LogisticRegressionGD(lr=0.1, n_epochs=2)
    model.train(X, y)
    # prediction = model.predict([4, 5])
    # assert isinstance(prediction, float)


def test_logistic_regression_gd_train():
    import numpy as np

    # Set seed for reproducibility
    np.random.seed(0)

    # Generate class 0 data (centered at (2, 2))
    x0 = np.random.randn(5, 2) + np.array([2, 2])
    y0 = np.zeros(5) - 1

    # Generate class 1 data (centered at (6, 6))
    x1 = np.random.randn(5, 2) + np.array([6, 6])
    y1 = np.ones(5)

    # Combine the data
    X = np.vstack((x0, x1)).tolist()
    y = np.concatenate((y0, y1)).tolist()
    model = LogisticRegressionGD(lr=0.1, n_epochs=100)
    result = model.train(X, y)
    assert "weight" in result
    assert "bias" in result
    assert isinstance(result["weight"], list)
    assert isinstance(result["bias"], float)


def test_logistic_regression_gd_predict():
    X = [[1, 2], [2, 3], [3, 4]]
    y = [1, -1, 1]
    model = LogisticRegressionGD(lr=0.1, n_epochs=100)
    model.train(X, y)
    prediction = model.predict([[4, 5]])
    assert isinstance(prediction, list)
    assert isinstance(prediction[0], int)


def test_logistic_regression_gd_init_weights():
    model = LogisticRegressionGD(lr=0.01, n_epochs=10)
    model._init_weights(3)  # type: ignore
    assert model.W == [0.0, 0.0, 0.0]
    assert model.b == 0


def test_logistic_regression_gd_training_step():
    model = LogisticRegressionGD(lr=0.01, n_epochs=10)
    model._init_weights(2)  # type: ignore
    grad_w, grad_b = model.training_step([1, 2], 1)
    assert len(grad_w) == 2
    assert isinstance(grad_b, float)
