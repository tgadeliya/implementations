from typing import Union, Iterable
from math import log2


def gini_impurity(x: Iterable[Union[str, int]]) -> float:
    """
    Calculate the Gini impurity of a list of classes.
    """
    sum_probs = 0
    for c in set(x):
        p = len([o for o in x if o == c]) / len(x)
        sum_probs += p**2
    return 1 - sum_probs


def shannon_entropy(x: Iterable[Union[str, int]]) -> float:
    """
    Calculate the Shannon entropy of a list of classes."""
    classes = set(x)
    sum_ent = 0
    for c in classes:
        p = [o for o in x if o == c] / len(x)
        sum_ent += -p * log2(p)
    return sum_ent


def missclassification_error(x: Iterable[Union[str, int]]) -> float:
    """
    Calculate the misclassification error of a list of classes.
    """
    probs = [len([o for o in x if o == c]) / len(x) for c in set(x)]
    return 1 - max(probs)
