import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from configs.path import CIFARPROCESSED, IMAGENET32PROCESSED, CAR196PROCESSED
from configs.info import CIFARNCLASS, IMAGENETNCLASS, CAR196NCLASS

from utils.reader_op import read_npy, read_pkl
from utils.datamanager import DATAMANAGER_DICT

import numpy as np

def cifar_manager(dm_type='basic', nsclass=0):
    '''
    Args:
        dm_type - string 
            defaults to be basic
        nsclass - int
            which required when dm_type='triplet'
    Return:
        dm_train, dm_val, dm_test
            datamanager for each set
    '''
    assert dm_type in DATAMANAGER_DICT.keys(), "The type of data should be in {}".format(DATAMANAGER_DICT.keys())

    dm = DATAMANAGER_DICT[dm_type]

    train_input = read_npy(CIFARPROCESSED+'train_image.npy')
    train_label = read_npy(CIFARPROCESSED+'train_label.npy')
    val_input = read_npy(CIFARPROCESSED+'val_image.npy')
    val_label = read_npy(CIFARPROCESSED+'val_label.npy')
    test_input = read_npy(CIFARPROCESSED+'test_image.npy')
    test_label = read_npy(CIFARPROCESSED+'test_label.npy')

    if dm_type in ['triplet', 'npair']: 
        dm_train = dm(train_input, train_label, CIFARNCLASS, nsclass) 
        dm_val = dm(val_input, val_label, CIFARNCLASS, nsclass)
        dm_test = dm(test_input, test_label, CIFARNCLASS, nsclass) 
    else:
        dm_train = dm(train_input, train_label, CIFARNCLASS)    
        dm_val = dm(val_input, val_label, CIFARNCLASS)    
        dm_test = dm(test_input, test_label, CIFARNCLASS)    
    return dm_train, dm_val, dm_test

def imgnet32_manager(dm_type='basic', nsclass=0):
    '''
    Args:   
        dm_type - string
        nsclass -int
    '''
    train_img = read_npy(IMAGENET32PROCESSED+'train_img.npy')
    train_label = read_npy(IMAGENET32PROCESSED+'train_label.npy')
    val_img = read_npy(IMAGENET32PROCESSED+'val_img.npy')
    val_label = read_npy(IMAGENET32PROCESSED+'val_label.npy')
    test_img = read_npy(IMAGENET32PROCESSED+'test_img.npy')
    test_label = read_npy(IMAGENET32PROCESSED+'test_label.npy')

    dm = DATAMANAGER_DICT[dm_type]
    if dm_type in ['triplet', 'npair']: 
        dm_train = dm(train_img, train_label, IMAGENETNCLASS, nsclass) 
        dm_val = dm(val_img, val_label, IMAGENETNCLASS, nsclass)
        dm_test = dm(test_img, test_label, IMAGENETNCLASS, nsclass) 
    else:
        dm_train = dm(train_img, train_label, IMAGENETNCLASS)    
        dm_val = dm(val_img, val_label, IMAGENETNCLASS)    
        dm_test = dm(test_img, test_label, IMAGENETNCLASS)    
    return dm_train, dm_val, dm_test

def car196_manager(dm_type='basic', nsclass=0):
    train_img = read_npy(path=CAR196PROCESSED+"train_img.npy")
    train_label = read_npy(path=CAR196PROCESSED+"train_label.npy")
    val_img = read_npy(path=CAR196PROCESSED+"test_img.npy")
    val_label = read_npy(path=CAR196PROCESSED+"test_label.npy")
    test_img = read_npy(path=CAR196PROCESSED+"test_img.npy")
    test_label = read_npy(path=CAR196PROCESSED+"test_label.npy")

    dm = DATAMANAGER_DICT[dm_type]
    if dm_type in ['triplet', 'npair']: 
        dm_train = dm(train_img, train_label, CAR196NCLASS, nsclass) 
        dm_val = dm(val_img, val_label, CAR196NCLASS, nsclass)
        dm_test = dm(test_img, test_label, CAR196NCLASS, nsclass) 
    else:
        dm_train = dm(train_img, train_label, CAR196NCLASS)    
        dm_val = dm(val_img, val_label, CAR196NCLASS)    
        dm_test = dm(test_img, test_label, CAR196NCLASS)    
    return dm_train, dm_val, dm_test

