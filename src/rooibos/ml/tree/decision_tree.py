from typing import Optional, Union, List, Tuple, Iterable, Set
from math import log2
from collections import Counter


# Popular criterions for decision tree split


def gini_impurity(x: Iterable[Union[str, int]]) -> float:
    """
    Calculate the Gini impurity of a list of classes.
    """
    sum_probs = 0
    total = len(x)
    for c in set(x):
        p = len([o for o in x if o == c]) / total
        sum_probs += p**2
    return 1 - sum_probs


def shannon_entropy(x: Iterable[Union[str, int]]) -> float:
    """
    Calculate the Shannon entropy of a list of classes."""
    classes = set(x)
    total = len(x)
    sum_ent = 0
    for c in classes:
        p = len([o for o in x if o == c]) / total
        sum_ent += -p * log2(p)
    return sum_ent


def missclassification_error(x: Iterable[Union[str, int]]) -> float:
    """
    Calculate the misclassification error of a list of classes.
    """
    probs = []
    total = len(x)
    for c in set(x):
        p = len([o for o in x if o == c]) / total
        probs.append(p)
    return 1 - max(probs)


class DecisionTreeNode:
    def __init__(self, parent: "DecisionTreeNode", depth: int):
        self.parent = parent
        self.depth = depth
        self.left = None
        self.right = None
        self.is_leaf = False
        self.split = {"feature_idx": None, "threshold": None}

    def set_split(self, feature_idx, threshold):
        self.split["feature_idx"] = feature_idx
        self.split["threshold"] = threshold


class DecisionTreeClassifier:
    """
    A Decision Tree Classifier.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        max_features: Optional[int] = None,
    ):
        self.criterion = self.get_criterion(criterion)
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features

        self.tree = None

    def get_criterion(self, criterion: str):
        """
        Get the criterion function based on the specified criterion.
        """
        if criterion == "gini":
            return gini_impurity
        elif criterion == "shannon_entropy":
            return shannon_entropy
        elif criterion == "misclassification_error":
            return missclassification_error
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def train(self, X, y):
        """
        Fit the decision tree classifier to the training data.
        """

        self.classes = set(y)
        self.n_classes = len(self.classes)

        def build_tree(self, parent_node, X, y, depth, used_features_idxs: Set[int]):
            if depth >= self.max_depth or used_features_idxs == self.max_features:
                return None

            # TODO: Add heuristic for too few samples

            if len(set(y)) == 1:
                return None

            node = DecisionTreeNode(parent=parent_node, depth=depth)
            # find best split for current node
            (best_feat_idx, best_threshold, left_idxs, right_idxs) = (
                self.get_best_split(X, y, used_features_idxs, self.criterion)
            )

            node.set_split(feature_idx=best_feat_idx, threshold=best_threshold)
            used_features_idxs.update(best_feat_idx)

            node.left = build_tree(
                node, X[left_idxs], y[left_idxs], depth + 1, used_features_idxs
            )
            node.right = build_tree(
                node, X[right_idxs], y[right_idxs], depth + 1, used_features_idxs
            )

            if node.left is None and node.right is None:  # leaf node
                node.split["class"] = Counter(y).most_common(1)[0][0]
                node.is_leaf = True
            return node

        self.tree = build_tree(None, X, y, 0, set())

    def get_best_split(self, X, y, used_features, criterion_func):
        available_features = [i for i in range(len(X[0])) if i not in used_features]
        inf_gains = []
        for feature_idx in available_features:
            ig_best, ig_best_thr, left_idxs_best, right_idxs_best = (
                self.get_information_gain(
                    X[feature_idx], y, feature_idx, criterion_func
                )
            )
            inf_gains.append(
                (feature_idx, ig_best, ig_best_thr, left_idxs_best, right_idxs_best)
            )

        inf_gains.sort(key=lambda x: x[1], reverse=True)
        best_feature_idx, best_ig, best_thr, left_idxs_best, right_idxs_best = (
            inf_gains[0]
        )
        return best_feature_idx, best_thr, left_idxs_best, right_idxs_best

    def get_information_gain(self, x, y, criterion_func):

        # TODO: Add support for categorical features
        xs = sorted(set(x))
        # TODO: Implement heurstic from mlcourse
        # Conclusion: the simplest heuristics for handling numeric features
        # in a decision tree is to sort its values in ascending order and check only those thresholds where the value of the target variable changes.
        thresholds = [(p[0] + p[1]) / 2 for p in zip(xs[:-1], xs[1:])]
        ig = criterion_func(y)
        ig_best, ig_best_thr = 0, None
        left_idxs_best, right_idxs_best = None, None
        for thr in thresholds:
            left = [i for i in range(len(x)) if i <= thr]
            right = [i for i in range(len(x)) if i > thr]
            ig_left = criterion_func(y[left])
            ig_right = criterion_func(y[right])
            ig_thr = (
                ig - (len(left) / len(x)) * ig_left - (len(right) / len(x)) * ig_right
            )

            if ig_thr > ig_best:
                ig_best = ig_thr
                ig_best_thr = thr
                left_idxs_best = left
                right_idxs_best = right
        return ig_best, ig_best_thr, left_idxs_best, right_idxs_best

    def predict(self, X: List[List[Union[int, float]]]) -> List[int]:
        """
        Predict the class labels for the input data.
        """
        return [self.predict_example(x) for x in X]

    def predict_example(self, x: List[Union[int, float]]) -> int:
        """
        Predict the class label for a single example.
        """
        node = self.tree
        while not node.is_leaf:
            f_idx, threshold = node.split["feature_idx"], node.split["threshold"]
            node = node.left if x[f_idx] <= threshold else node.right
        return node.split["class"]


# TODO: Add decision tree regressor
