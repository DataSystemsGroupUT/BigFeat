"""
tree_utils.py
-------------
Responsible for extracting paths and feature combinations from decision trees.
"""

from sklearn.tree import _tree


def get_paths(clf, feature_names):
    """ Returns every path in the decision tree"""
    tree_ = clf.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    path = []
    path_list = []

    def recurse(node, depth, path_list):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            path_list.append(path.copy())
        else:
            name = feature_name[node]
            path.append(name)
            recurse(tree_.children_left[node], depth + 1, path_list)
            recurse(tree_.children_right[node], depth + 1, path_list)
            path.pop()

    recurse(0, 1, path_list)

    new_list = []
    for i in range(len(path_list)):
        if path_list[i] != path_list[i - 1]:
            new_list.append(path_list[i])
    return new_list


def get_combos(paths, comb_mat):
    """ Fills Combination matrix with values """
    for i in range(len(comb_mat)):
        for pt in paths:
            if i in pt:
                comb_mat[i][pt] += 1


def get_split_feats(paths, split_vec):
    """ Fills split vector with values """
    for i in range(len(split_vec)):
        for pt in paths:
            if i in pt:
                split_vec[i] += 1