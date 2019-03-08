import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from configs.path import subdirs5resultdir, muldir2mulsubdir
from configs.info import K_SET

from utils.datasetmanager import cifar_manager
from utils.format_op import FileIdManager
from utils.writer_op import write_pkl, create_muldir
from utils.shutil_op import remove_dir, copy_dir
from utils.reader_op import read_pkl
from utils.csv_op import write_dict_csv

from local_config import local_cifar_parser, RESULT_DIR, ID_STRUCTURE

from model import Model

import numpy as np 

import itertools
import glob
import os 

def train_model():
    '''Run for several hyper parameters'''
    EPOCH = 300
    parser = local_cifar_parser()
    args = parser.parse_args() 

    fim = FileIdManager(ID_STRUCTURE)
    FILE_ID = fim.get_id_from_args(args)
    SAVE_DIR, LOG_DIR, CSV_DIR, PKL_DIR, BOARD_DIR, ASSET_DIR = subdirs5resultdir(RESULT_DIR, True)
    SAVE_SUBDIR, PKL_SUBDIR, BOARD_SUBDIR, ASSET_SUBDIR, CSV_SUBDIR = muldir2mulsubdir([SAVE_DIR, PKL_DIR, BOARD_DIR, ASSET_DIR, CSV_DIR], FILE_ID, True)

    # load data
    dm_train, dm_val, dm_test = cifar_manager(dm_type=args.ltype, nsclass=args.nsclass) 
    for v in [dm_train, dm_val, dm_test]: v.print_shape()

    model = Model(dm_train, dm_val, dm_test, LOG_DIR+FILE_ID+'.log', args)
    model.build()
    model.set_up_train()
    try:
        model.restore(save_dir=SAVE_SUBDIR)
    except (AttributeError, TypeError):
        model.initialize()
        model.train(epoch=EPOCH, save_dir=SAVE_SUBDIR, board_dir=BOARD_SUBDIR)
        model.restore(save_dir=SAVE_SUBDIR)
    model.prepare_test()
    content = model.test_metric(K_SET)
    write_pkl(content, path=PKL_SUBDIR+'evaluation.pkl')
    write_dict_csv(dict_=content, path=CSV_SUBDIR+'evaluation.csv')

def integrate_results_and_preprocess():
    SAVE_DIR, LOG_DIR, CSV_DIR, PKL_DIR, BOARD_DIR, ASSET_DIR = subdirs5resultdir(RESULT_DIR, False)
    BOUNDARY = 10 
    FILE_KEY = 'evaluation.pkl'
    def get_value(path):
        content = read_pkl(path)
        return np.sum(content['te_te_precision_at_k'])
    max_value = -1
    max_file = None
    for file in sorted(os.listdir(PKL_DIR)):
        PKL_SUBDIR = PKL_DIR+'{}/'.format(file)
        path = PKL_SUBDIR+'{}'.format(FILE_KEY)
        if os.path.exists(path):
            value = get_value(path)
            if max_value<value:
                max_value=value
                max_file = file
    # Get the best file id
    parser = local_cifar_parser()
    args = parser.parse_args() 
    fim = FileIdManager(ID_STRUCTURE)
    fim.update_args_with_id(args, max_file) # update args value
    FILE_ID = fim.get_id_from_args(args)

    SAVE_SUBDIR, PKL_SUBDIR, BOARD_SUBDIR, ASSET_SUBDIR, CSV_SUBDIR = muldir2mulsubdir([SAVE_DIR, PKL_DIR, BOARD_DIR, ASSET_DIR, CSV_DIR], FILE_ID, False)
    # load data
    dm_train, dm_val, dm_test = cifar_manager(dm_type=args.ltype, nsclass=args.nsclass) 
    for v in [dm_train, dm_val, dm_test]: v.print_shape()

    model = Model(dm_train, dm_val, dm_test, './test.log', args)
    model.build()
    model.set_up_train()
    model.restore(save_dir=SAVE_SUBDIR)
    model.prepare_test()
    model.prepare_test2()
    META_DIR = RESULT_DIR+'meta/'
    BESTSAVE_DIR = RESULT_DIR+'bestsave/'

    # copy file 
    if os.path.isdir(BESTSAVE_DIR): remove_dir(BESTSAVE_DIR)
    copy_dir(SAVE_SUBDIR, BESTSAVE_DIR)
    # ======================================================#
    create_muldir(META_DIR)
    store_content = {
            'train_embed' : model.train_embed,
            'test_embed' : model.test_embed,
            'val_embed' : model.val_embed,
            'te_te_distance' : model.te_te_distance,
            'te_tr_distance' : model.te_tr_distance,
            'val_arg_sort' : model.val_arg_sort,
            }
    for v in store_content.keys():
        print("{} : {}".format(v, store_content[v].shape))
    write_pkl(store_content, META_DIR+'meta.pkl')

if __name__ == '__main__':
    #train_model()
    integrate_results_and_preprocess()
