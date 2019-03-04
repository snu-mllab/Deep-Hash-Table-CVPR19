import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.eval_op import get_recall_precision_at_k, get_recall_precision_at_k_arg,\
                        get_VAL_at_FAR, get_DIR_at_FAR, get_rank_k,\
                        get_VAL_at_FAR2, get_DIR_at_FAR2,\
                        HashTable, HashTree, HashTableV2
from utils.format_op import listformat
import numpy as np

def evaluate_metric_te_tr(test_label, train_label, te_te_distance, te_tr_distance, k_set, logger=None):
    '''
    Args:
        test_label - Numpy 1D array [ntest]
        train_label - Numpy 1D array [ntrain]
        te_te_distance - Numpy 2D array [ntest, ntest]
        te_tr_distance - Numpy 2D array [ntest, ntrain]
        k_set - list
        logger - logger
            defaults to be None
    Return:
        performance - dict
    '''
    te_te_recall_at_k, te_te_precision_at_k = get_recall_precision_at_k(dist_matrix=te_te_distance, labelq=test_label, labelh=test_label, k_set=k_set, issame=True)
    te_tr_recall_at_k, te_tr_precision_at_k = get_recall_precision_at_k(dist_matrix=te_tr_distance, labelq=test_label, labelh=train_label, k_set=k_set, issame=False)

    performance = {
                'te_tr_precision_at_k' : te_tr_precision_at_k,
                'te_te_precision_at_k' : te_te_precision_at_k,
                'te_tr_recall_at_k' : te_tr_recall_at_k,
                'te_te_recall_at_k' : te_te_recall_at_k}
    key_set = ['te_tr_recall_at_k', 'te_tr_precision_at_k', 'te_te_recall_at_k', 'te_te_precision_at_k']
           
    for key in key_set:
        content = '{} @ {} =  {}'.format(str(key), listformat(k_set), listformat(performance[key]))
        if logger is None:
            print(content)
        else:
            logger.info(content)

    return performance

def evaluate_hashtree_te_tr_sparsity(te_tr_distance, te_te_distance, train_tree_idx, test_tree_idx, train_max_k_idx, test_max_k_idx, train_label, test_label, ncls_train, ncls_test, k_set, logger=None):
    '''
    Args:
        te_tr_distance - Numpy 2D array [ntest, ntrain]
        te_te_distance - Numpy 2D array [ntest, ntest]
        train_tree_idx - Numpy 2D array [ntrain, k, d]
        test_tree_idx - Numpy 2D array [ntest, k, d]
        train_max_k_idx - Numpy 2D array [ntrain, sparsity]
        test_max_k_idx - Numpy 2D array [ntest, sparsity]
        train_label - label of train data [ntrain]
        test_label - label of test data [ntest] 
        ncls_train - int, number of train classes or labels
        ncls_test - int, number of test classes or labels
        k_set - list [nk]
        logger - logger
            defaults to be None
    Return: 
        performance - dict
    '''
    _, depth, width = train_tree_idx.shape
    train_hashtree = HashTree(depth=depth, width=width, max_k_idx=train_max_k_idx, labelh=train_label, nlabel=ncls_train) 
    test_hashtree = HashTree(depth=depth, width=width, max_k_idx=test_max_k_idx, labelh=test_label, nlabel=ncls_test) 

    te_te_srr, te_te_recall_at_k, te_te_precision_at_k = test_hashtree.get_srr_recall_precision_at_k_hash(dist_matrix=te_te_distance, query_tree_idx=test_tree_idx, query_max_k_idx=test_max_k_idx, labelq=test_label, k_set=k_set, issame=True)
    te_tr_srr, te_tr_recall_at_k, te_tr_precision_at_k = train_hashtree.get_srr_recall_precision_at_k_hash(dist_matrix=te_tr_distance, query_tree_idx=test_tree_idx, query_max_k_idx=test_max_k_idx, labelq=test_label, k_set=k_set, issame=False)
    
    te_tr_suf = 1.0/np.mean(te_tr_srr, axis=-1)
    te_te_suf = 1.0/np.mean(te_te_srr, axis=-1)

    performance = {
            'train_nmi' : train_hashtree.nmi,
            'test_nmi' : test_hashtree.nmi,
            'te_tr_suf' : te_tr_suf,
            'te_te_suf' : te_te_suf,
            'te_tr_precision_at_k' : te_tr_precision_at_k,
            'te_te_precision_at_k' : te_te_precision_at_k,
            'te_tr_recall_at_k' : te_tr_recall_at_k,
            'te_te_recall_at_k' : te_te_recall_at_k
            }

    key_set = [
        'train_nmi',
        'test_nmi',
        'te_tr_suf',
        'te_te_suf',
        'te_tr_precision_at_k',
        'te_te_precision_at_k',
        'te_tr_recall_at_k',
        'te_te_recall_at_k'
        ]

    for key in key_set:
        if 'suf' in str(key): content = "{} {} = {}".format(str(key), listformat(k_set), listformat(performance[key])) 
        elif 'at_k' in str(key): content = "{} {} = {}".format(str(key), listformat(k_set), listformat(performance[key])) 
        else: content = "{} = {:.4f}".format(str(key), performance[key])
        
        if logger is None: print(content)
        else: logger.info(content)

    return performance

