
def gini_index(groups, classes):
    n = sum([len(g) for g in groups])
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val)/ size
            score += p * p
        gini += (1 - score) * (size / n)
    return gini

def test_split(index, value, dataset):
    l, r = [], []
    for row in dataset:
        if row[index] < value:
            l.append(row)
        else:
            r.append(row)
    return l, r

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_score, b_value, b_groups = 999, float('inf'), 999, None
    for index in range(len(dataset[0])-1): # last column as target
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {"index":b_index, "value": b_value, "groups": b_groups}

def create_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def recurse_split(node, max_depth, min_size, depth):
    left, right = node["groups"]
    del(node["groups"])

    if not left or not right:
        node["left"] = create_terminal(left + right)
        return 
    
    if depth >= max_depth:
        node["left"], node["right"] = create_terminal(left), create_terminal(right)
        return 

    if len(left) <= min_size:
        node["left"] = create_terminal(left)
    else:
        node["left"] = get_split(left)
        recurse_split(node["left"], max_depth, min_size, depth+1)

    if len(right) <= min_size:
        node["right"] = create_terminal(right)
    else:
        node["right"] = get_split(right)
        recurse_split(node["right"], max_depth, min_size, depth+1)


def build_tree(train, max_depth, min_size):
    root = get_split(train)
    recurse_split(root, max_depth, min_size, 1)
    return root




# def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
#     pass



if __name__ == "__main__":
    dataset = [
        [18, 1, 0],
        [20, 0, 1],
        [23, 2, 1],
        [25, 1, 1],
        [30, 1, 0],
    ]

    split = get_split(dataset)
    print('\nBest Split:')
    print('Column Index: %s, Value: %s' % ((split['index']), (split['value'])))
    # Output: Column Index: 0, Value: 20


    # Sample dataset
    dataset = [
        [5, 3, 0], [6, 3, 0], [6, 4, 0], [10, 3, 1],
        [11, 4, 1], [12, 8, 0], [5, 5, 0], [12, 4, 1]
    ]

    max_depth = 2
    min_size = 1
    tree = build_tree(dataset, max_depth, min_size)

    # Print the tree
    def print_tree(node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
            print_tree(node['left'], depth+1)
            print_tree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', node)))

    print_tree(tree)
    '''Output:
    [X1 < 10.000]
    [X1 < 5.000]
    [0]
    [0]
    [X2 < 8.000]
    [1]
    [0]
    '''