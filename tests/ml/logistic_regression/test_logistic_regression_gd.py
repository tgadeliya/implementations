import pytest
from rooibos.ml.logistic_regression.logistic_regression_gd import LogisticRegressionGD

def test_logistic_regression_gd_init():
    LogisticRegressionGD(lr=0.01, n_epochs=10)

def test_logistic_regression_gd_train_small():
    X = [[1, 2], [2, 3], [3, 4]]
    y = [1, 1, -1]
    model = LogisticRegressionGD(lr=0.1, n_epochs=2)
    model.train(X, y)
    # prediction = model.predict([4, 5])
    # assert isinstance(prediction, float)

def test_logistic_regression_gd_train():
    import random
    random.seed(0)

    x0 = [[random.gauss(0, 1) + 2, random.gauss(0, 1) + 2] for _ in range(5)]
    y0 = [-1] * 5

    x1 = [[random.gauss(0, 1) + 6, random.gauss(0, 1) + 6] for _ in range(5)]
    y1 = [1] * 5

    X = x0 + x1
    y = y0 + y1
    model = LogisticRegressionGD(lr=0.1, n_epochs=100)
    result = model.train(X, y)
    assert "weight" in result
    assert "bias" in result
    assert isinstance(result["weight"], list)
    assert isinstance(result["bias"], float)

def test_logistic_regression_gd_predict():
    X = [[1, 2], [2, 3], [3, 4]]
    y = [1, 0, 1]
    model = LogisticRegressionGD(lr=0.1, n_epochs=100)
    model.train(X, y)
    prediction = model.predict([4, 5])
    assert isinstance(prediction, float)

def test_logistic_regression_gd_init_weights():
    model = LogisticRegressionGD(lr=0.01, n_epochs=10)
    model._init_weights(3)
    assert model.W == [0.0, 0.0, 0.0]
    assert model.b == 0

def test_logistic_regression_gd_training_step():
    model = LogisticRegressionGD(lr=0.01, n_epochs=10)
    model._init_weights(2)
    x = [1, 2]
    y = 1
    grad_w, grad_b = model.training_step(x, y)
    assert len(grad_w) == 2
    assert isinstance(grad_b, float)