from collections import Counter
from math import log2

# Popular criterions for decision tree split


def gini_impurity(x: list[str | int]) -> float:
    """Calculate the Gini impurity of a list of classes."""
    sum_probs: float = 0.0
    for c in set(x):
        p: float = len([o for o in x if o == c]) / len(x)
        sum_probs += p**2
    return 1 - sum_probs


def shannon_entropy(x: list[str | int]) -> float:
    """Calculate the Shannon entropy of a list of classes."""
    classes = set(x)
    sum_ent = 0.0
    for c in classes:
        p = len([o for o in x if o == c]) / len(x)
        sum_ent += -p * log2(p)
    return sum_ent


def missclassification_error(x: list[str | int]) -> float:
    """Calculate the misclassification error of a list of classes."""
    probs: list[float] = []
    for c in set(x):
        p: float = len([o for o in x if o == c]) / len(x)
        probs.append(p)
    return 1 - max(probs)


class DecisionTreeNode:
    def __init__(self, parent: "DecisionTreeNode", depth: int):
        self.parent = parent
        self.depth = depth
        self.left = None
        self.right = None
        self.is_leaf = False
        self.split = {"feature_idx": None, "threshold": None, "class": None}

    def set_split(self, feature_idx: int, threshold: float) -> None:
        self.split["feature_idx"] = feature_idx
        self.split["threshold"] = threshold


class DecisionTreeClassifier:
    """A Decision Tree Classifier."""

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        max_leaf_nodes: int | None = None,
        max_features: int | None = None,
        max_samples_per_node: int | None = None,
    ):
        self.criterion = self.get_criterion(criterion)
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.max_samples_per_node = max_samples_per_node

        self.tree: DecisionTreeNode
        self.labels: set[float | str]

    def get_criterion(self, criterion: str):
        """Get the criterion function based on the specified criterion."""
        if criterion == "gini":
            return gini_impurity
        elif criterion == "shannon_entropy":
            return shannon_entropy
        elif criterion == "misclassification_error":
            return missclassification_error
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def train(self, X: list[list[float]], y: list[float | str]) -> None:
        """Fit the decision tree classifier to the training data."""
        self.labels = set(y)
        
        def build_tree(
            parent_node: DecisionTreeClassifier | None,
            X: list[list[float]],
            y: list[float | str],
            depth: int,
            used_features_idxs: set[int],
        ) -> DecisionTreeNode | None:
            
            if self.max_depth and depth >= self.max_depth:
                return None  # max depth reached

            if self.max_features and len(used_features_idxs) == self.max_features:
                return None  # max features reached
            
            if len(set(y)) == 1:  # TODO: AND?
                return None  # all samples have the same class

            # TODO: Add heuristic for too few samples

            node = DecisionTreeNode(parent=parent_node, depth=depth)
            # find best split for current node
            (best_feat_idx, best_threshold, left_idxs, right_idxs) = (
                self.get_best_split(X, y, used_features_idxs, self.criterion)
            )

            node.set_split(feature_idx=best_feat_idx, threshold=best_threshold)
            used_features_idxs.add(best_feat_idx)

            X_left = [X[idx] for idx in left_idxs]
            y_left = [y[idx] for idx in left_idxs]
            X_right = [X[idx] for idx in right_idxs]
            y_right = [y[idx] for idx in right_idxs]
            node.left = build_tree(node, X_left, y_left, depth + 1, used_features_idxs)
            node.right = build_tree(
                node, X_right, y_right, depth + 1, used_features_idxs
            )

            if node.left is None and node.right is None:  # leaf node
                node.is_leaf = True
            node.split["class"] = Counter(y).most_common(1)[0][0]
            return node

        self.tree = build_tree(None, X, y, 0, set())  # TODO: check depth 0 is ok?

    def get_best_split(self, X, y, used_features, criterion_func):
        available_features = [
            i for i in range(len(X[0])) if i not in used_features
        ]
        inf_gains = []
        for feature_idx in available_features:
            ig_best, ig_best_thr, left_idxs_best, right_idxs_best = (
                self.get_information_gain(
                    [x[feature_idx] for x in X], y, criterion_func
                )
            )
            inf_gains.append(
                (
                    feature_idx,
                    ig_best,
                    ig_best_thr,
                    left_idxs_best,
                    right_idxs_best,
                )
            )

        inf_gains.sort(key=lambda x: x[1], reverse=True)
        best_feature_idx, best_ig, best_thr, left_idxs_best, right_idxs_best = (
            inf_gains[0]
        )
        return best_feature_idx, best_thr, left_idxs_best, right_idxs_best

    def get_information_gain(self, x, y, criterion_func):

        # Top-N heuristic
        # Furthermore, when there are a lot of numeric features
        # in a dataset, each with a lot of unique values, only the
        # top-N of the thresholds described above are selected, i.e.
        # only use the top-N that give maximum gain.
        # The process is to construct a tree of depth 1, compute the
        #  entropy (or Gini uncertainty),
        #  and select the best thresholds for comparison.

        # TODO: Add support for categorical features
        xs = sorted(set(x))
        # TODO: Implement heurstic from mlcourse
        # Conclusion: the simplest heuristics for handling numeric features
        # in a decision tree is to sort its values in ascending order and
        # check only those thresholds where the value of the target
        # variable changes.
        thresholds = [(p[0] + p[1]) / 2 for p in zip(xs[:-1], xs[1:])]
        ig = criterion_func(y)
        ig_best, ig_best_thr = 0, None
        left_idxs_best, right_idxs_best = None, None
        for thr in thresholds:
            left = [i for i in range(len(x)) if i <= thr]
            right = [i for i in range(len(x)) if i > thr]
            yl = [y[i] for i in left]
            yr = [y[i] for i in right]
            ig_left = criterion_func(yl)
            ig_right = criterion_func(yr)
            ig_thr = (
                ig
                - (len(left) / len(x)) * ig_left
                - (len(right) / len(x)) * ig_right
            )

            if ig_thr > ig_best:
                ig_best = ig_thr
                ig_best_thr = thr
                left_idxs_best = left
                right_idxs_best = right
        return ig_best, ig_best_thr, left_idxs_best, right_idxs_best

    def predict(self, X: list[list[int | float]]) -> list[int]:
        """Predict the class labels for the input data."""
        return [self.predict_example(x) for x in X]

    def predict_example(self, x: List[Union[int, float]]) -> int:
        """
        Predict the class label for a single example.
        """
        node = self.tree
        pred = node.split["class"]
        while node and (not node.is_leaf):
            f_idx, threshold, pred = (
                node.split["feature_idx"],
                node.split["threshold"],
                node.split["class"],
            )
            node = node.left if x[f_idx] <= threshold else node.right
        return pred


# TODO: Add decision tree regressor
