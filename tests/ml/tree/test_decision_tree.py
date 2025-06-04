import math
import pytest
from rooibos.ml.tree.decision_tree import (
    gini_impurity,
    shannon_entropy,
    missclassification_error,
    DecisionTreeClassifier,
)


def test_gini_impurity():
    assert math.isclose(gini_impurity([0, 1, 0, 1]), 0.5)


def test_shannon_entropy():
    ent = shannon_entropy([0, 1, 0, 1])
    assert math.isclose(ent, 1.0, rel_tol=1e-5)


def test_misclassification_error():
    err = missclassification_error([0, 1, 0, 1])
    assert math.isclose(err, 0.5)


def test_get_criterion_valid():
    for name, fn in [
        ("gini", gini_impurity),
        ("shannon_entropy", shannon_entropy),
        ("misclassification_error", missclassification_error),
    ]:
        clf = DecisionTreeClassifier(criterion=name)
        assert clf.criterion is fn


def test_get_criterion_invalid():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="unknown")
