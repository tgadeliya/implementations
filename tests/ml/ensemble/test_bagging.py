import pytest
from rooibos.ml.ensemble.bagging import BaggingClassifier, BaggingRegressor
from rooibos.ml.linear_regression.linear_regression_gd import LinearRegressionSGD

# filepath: src/rooibos/ml/ensemble/test_bagging.py


def test_bagging_init():
    model1 = LinearRegressionSGD(eta=0.01, n_epochs=10)
    model2 = LinearRegressionSGD(eta=0.01, n_epochs=10)
    models = [model1, model2]

    bagging = BaggingRegressor(models=models, bootstrap_size=0.8)
    assert bagging.models == models
    assert bagging.bootstrap_size == 0.8
    assert bagging.n_models == 2

    class InvalidModel:
        pass

    with pytest.raises(AttributeError):
        BaggingRegressor(models=[InvalidModel()], bootstrap_size=0.8)  # Missing methods


def test_bagging_regressor_with_linear_regression():
    # Small dataset
    X = [[1], [2], [3], [4]]
    y = [2, 4, 6, 8]

    # Create LinearRegressionSGD models
    model1 = LinearRegressionSGD(eta=0.01, n_epochs=100)
    model2 = LinearRegressionSGD(eta=0.01, n_epochs=100)
    models = [model1, model2]

    # Initialize BaggingRegressor
    bagging = BaggingRegressor(models=models, bootstrap_size=0.8, aggregation="mean")

    # Train the BaggingRegressor
    bagging.train(X, y)

    # Predict on new data
    predictions = bagging.predict([[5], [6]])
    assert len(predictions) == 2
    assert all(isinstance(pred, float) for pred in predictions)