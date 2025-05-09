from math import log2

MAX_DEPTH = float("inf")
MIN_SIZE = 1


def entropy(X):
    n = len(X)
    ent = 0
    for x in set(X):
        xs = X.count(x)
        p = xs / n
        ent += p * log2(p)
    return -ent


def s2i(examples, attr):
    d = {}
    for a in attr:
        a_ex = set([e[a] for e in examples])
        d[a] = dict(zip(a_ex, range(len(a_ex))))
    return d


def preprocess_data(examples, attr, target):
    prep_ex = []
    d = s2i(examples, attr)
    ys = []
    for row in examples:
        ex = [d[k][v] for k, v in row.items() if k in attr]
        ys.append(row[target])
        prep_ex.append(ex)
    return prep_ex, ys


def split_dataset(f_idx, thr, dataset, y):
    l, r, yl, yr = [], [], [], []
    for row, row_y in zip(dataset, y):
        if row[f_idx] == thr:
            l.append(row)
            yl.append(row_y)
        else:
            r.append(row)
            yr.append(row_y)
    return l, r, yl, yr


def calculate_information_gain(y, yl, yr, criterion=entropy):
    assert len(y) == len(yl) + len(yr), "length in splits do not merge"
    ylen = len(y)
    return (
        criterion(y)
        - criterion(yl) * (len(yl) / ylen)
        - criterion(yr) * (len(yr) / ylen)
    )


def best_gini_split(dataset, y):
    best_feat, best_thr = -1, -1
    best_score, best_l, best_r, best_yr, best_yl = float("-inf"), None, None, None, None
    for i in range(len(dataset[0])):
        thrs = set([d[i] for d in dataset])
        for thr in thrs:
            l, r, yl, yr = split_dataset(i, thr, dataset, y)
            score = calculate_information_gain(y, yl, yr)
            if best_score < score:
                best_score = score
                best_feat, best_thr = i, thr
                best_l, best_r, best_yr, best_yl = l, r, yl, yr

    return best_l, best_r, best_yr, best_yl, best_feat, best_thr


def build_tree(X, target, depth, min_size=MIN_SIZE, max_depth=MAX_DEPTH):
    node = {}  # create node
    if depth > max_depth or len(X) <= min_size:
        node["is_terminal"] = True
        node["prediction"] = max(set(target), key=target.count)
    else:
        ldataset, rdataset, ltarget, rtarget, feat_idx, thr = best_gini_split(X, target)
        if len(ldataset) == 0 or len(rdataset) == 0:  # no sense to split futher
            node["is_terminal"] = True
            node["prediction"] = max(set(target), key=target.count)
        else:
            node["split"] = (feat_idx, thr)  # if val < thr then left
            node["left"] = build_tree(ldataset, ltarget, depth + 1)
            node["right"] = build_tree(rdataset, rtarget, depth + 1)
            node["is_terminal"] = False
    # if we stop in this node, what prediction will be made
    return node


def learn_decision_tree(
    examples: list[dict], attributes: list[str], target_attr: str
) -> dict:
    X, y = preprocess_data(examples, attributes, target_attr)
    root = build_tree(X, y, 0)
    return root


if __name__ == "__main__":
    examples = [
        {
            "Outlook": "Sunny",
            "Temperature": "Hot",
            "Humidity": "High",
            "Wind": "Weak",
            "PlayTennis": "No",
        },
        {
            "Outlook": "Sunny",
            "Temperature": "Hot",
            "Humidity": "High",
            "Wind": "Strong",
            "PlayTennis": "No",
        },
        {
            "Outlook": "Overcast",
            "Temperature": "Hot",
            "Humidity": "High",
            "Wind": "Weak",
            "PlayTennis": "Yes",
        },
        {
            "Outlook": "Rain",
            "Temperature": "Mild",
            "Humidity": "High",
            "Wind": "Weak",
            "PlayTennis": "Yes",
        },
    ]
    attributes = ["Outlook", "Temperature", "Humidity", "Wind"]
    target = "PlayTennis"

    tree = learn_decision_tree(examples, attributes, target)
