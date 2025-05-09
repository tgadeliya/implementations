import pytest
from rooibos.ml.linear_regression import LinearRegressionSGD



def test_linear_regression_sgd_init():
    LinearRegressionSGD(eta=0.01, n_epochs=10)


def test_linear_regression_sgd_train():
    X = [[1], [2], [3]]
    y = [2, 3, 4]
    model = LinearRegressionSGD(eta=0.1, n_epochs=300)
    result = model.train(X, y)
    assert "weight" in result
    assert "bias" in result
    assert isinstance(result["weight"], list)
    assert isinstance(result["bias"], float)

def test_linear_regression_sgd_predict():
    X = [[1, 2], [2, 3], [3, 4]]
    y = [5, 7, 9]
    model = LinearRegressionSGD(eta=0.01, n_epochs=10)
    model.train(X, y)
    prediction = model.predict_example([4, 5])
    assert isinstance(prediction, float)

def test_linear_regression_sgd_init_weights():
    model = LinearRegressionSGD(eta=0.01, n_epochs=10)
    model.init_weights(3)
    assert model.W == [0.0, 0.0, 0.0]
    assert model.b == 0

def test_linear_regression_sgd_training_step():
    model = LinearRegressionSGD(eta=0.01, n_epochs=10)
    model.init_weights(2)
    x = [1, 2]
    y = 5
    grad_w, grad_b = model.training_step(x, y)
    assert len(grad_w) == 2
    assert isinstance(grad_b, float)