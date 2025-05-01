from pytest import fixture
from pytest import mark
from rooibos.ml import DecisionTreeClassifier


@fixture
def train_data():
    # Create a simple dataset
    X = [[0, 0], [1, 1], [2, 2], [3, 3]]
    y = [0, 1, 1, 0]
    return X, y



class TestDecisionTreeClassifier:

    def test_dry_run(self):
        clf = DecisionTreeClassifier()

    def test_dry_train(self, train_data):
        X, y = train_data
        clf = DecisionTreeClassifier()
        clf.train(X, y)
