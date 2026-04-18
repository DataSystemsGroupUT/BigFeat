"""
generator.py
------------
Responsible for recursively generating new features (Feature Crossing/Engineering).
"""

import numpy as np


def feat_with_depth(X, depth, op_ls, feat_ls, rng, ig_vector, operators, 
                    operator_weights, binary_operators, unary_operators):
    """ Recursively generate a new features """
    if depth == 0:
        feat_ind = rng.choice(np.arange(len(ig_vector)), p=ig_vector)
        feat_ls.append(feat_ind)
        return X[:, feat_ind]
    
    depth -= 1
    op = rng.choice(operators, p=operator_weights)
    
    if op in binary_operators:
        feat_1 = feat_with_depth(X, depth, op_ls, feat_ls, rng, ig_vector, 
                                 operators, operator_weights, binary_operators, unary_operators)
        feat_2 = feat_with_depth(X, depth, op_ls, feat_ls, rng, ig_vector, 
                                 operators, operator_weights, binary_operators, unary_operators)
        op_ls.append((op, depth))
        return op(feat_1, feat_2)
        
    elif op in unary_operators:
        feat_1 = feat_with_depth(X, depth, op_ls, feat_ls, rng, ig_vector, 
                                 operators, operator_weights, binary_operators, unary_operators)
        op_ls.append((op, depth))
        return op(feat_1)


def feat_with_depth_gen(X, depth, op_ls, feat_ls, binary_operators, unary_operators):
    """ Reproduce generated features with new data """
    if depth == 0:
        feat_ind = feat_ls.pop()
        return X[:, feat_ind]
    
    depth -= 1
    op = op_ls.pop()[0]
    
    if op in binary_operators:
        feat_1 = feat_with_depth_gen(X, depth, op_ls, feat_ls, binary_operators, unary_operators)
        feat_2 = feat_with_depth_gen(X, depth, op_ls, feat_ls, binary_operators, unary_operators)
        return op(feat_2, feat_1)
        
    elif op in unary_operators:
        feat_1 = feat_with_depth_gen(X, depth, op_ls, feat_ls, binary_operators, unary_operators)
        return op(feat_1)