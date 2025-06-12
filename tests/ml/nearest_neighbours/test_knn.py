import pytest

from rooibos.ml.nearest_neighbours import KNNClassifier, KNNRegressor


@pytest.mark.parametrize("model", [KNNClassifier, KNNRegressor])
def test_model_supported_metrics(model):
    m = model(n_neighbors=3, metric="euclidean")
    assert m.metric == "euclidean" and m.n_neighbors == 3


@pytest.mark.parametrize(
    "model, metric",
    [
        (KNNClassifier, "invalid_metric"),
        (KNNRegressor, "invalid_metric"),
    ],
)
def test_model_initialization_with_invalid_metrics(model, metric):
    with pytest.raises(AssertionError):
        model(n_neighbors=3, metric=metric)


def test_knn_classifier_prediction():
    model = KNNClassifier(n_neighbors=3, metric="euclidean")
    model.train([[1, 2], [2, 3], [3, 4]], [0, 1, 0])
    predictions = model.predict([[1.5, 2.5], [3, 3]])
    assert predictions == [0, 0]


def test_knn_regressor_prediction_mean():
    model = KNNRegressor(n_neighbors=2, metric="euclidean", aggregate="mean")
    model.train([[1, 2], [2, 3], [3, 4]], [1.0, 2.0, 3.0])
    predictions = model.predict([[1.5, 2.5], [3, 3]])
    assert predictions == [1.5, 2.5]


def test_knn_regressor_prediction_median():
    model = KNNRegressor(n_neighbors=2, metric="euclidean", aggregate="median")
    model.train([[1, 2], [2, 3], [3, 4]], [1.0, 2.0, 3.0])
    predictions = model.predict([[1.5, 2.5], [3, 3]])
    assert predictions == [1.5, 2.5]


def test_knn_regressor_invalid_aggregation():
    with pytest.raises(AssertionError):
        KNNRegressor(n_neighbors=3, metric="euclidean", aggregate="invalid_agg")
