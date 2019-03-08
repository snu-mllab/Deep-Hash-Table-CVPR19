import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from configs.path import subdirs5resultdir, muldir2mulsubdir
from configs.info import K_SET

from utils.datasetmanager import cifar_manager
from utils.reader_op import read_pkl
from utils.format_op import params2id, listformat, FileIdManager
from utils.csv_op import CsvWriter2, read_csv
from utils.writer_op import create_muldir, write_pkl

from local_config import RESULT_DIR, local_cifar_parser,\
                         ID_STRUCTURE

from model import Model

import numpy as np 
import argparse

def train_model():
    EPOCH = 300

    nk = len(K_SET)
    parser = local_cifar_parser()
    args = parser.parse_args() # parameter required for model

    fim = FileIdManager(ID_STRUCTURE)
    FILE_ID = fim.get_id_from_args(args)
    SAVE_DIR, LOG_DIR, CSV_DIR, PKL_DIR, BOARD_DIR, ASSET_DIR = subdirs5resultdir(RESULT_DIR, True)
    SAVE_SUBDIR, PKL_SUBDIR, BOARD_SUBDIR, ASSET_SUBDIR, CSV_SUBDIR = muldir2mulsubdir([SAVE_DIR, PKL_DIR, BOARD_DIR, ASSET_DIR, CSV_DIR], FILE_ID, True)

    dm_train, dm_val, dm_test = cifar_manager(dm_type=args.ltype, nsclass=args.nsclass) 
    for v in [dm_train, dm_val, dm_test]: v.print_shape()

    model = Model(dm_train, dm_val, dm_test, LOG_DIR+FILE_ID+'.log', args)
    model.build()
    meta = read_pkl(args.meta)
    model.restore(args.save)
    model.set_info(val_arg_sort=meta['val_arg_sort'], te_te_distance=meta['te_te_distance'], te_tr_distance=meta['te_tr_distance'])
    model.build_hash()
    model.set_up_train_hash()
    try:
        model.restore(save_dir=SAVE_SUBDIR)
    except (AttributeError, TypeError):
        model.initialize()
        model.train_hash(epoch=EPOCH, save_dir=SAVE_SUBDIR, board_dir=BOARD_SUBDIR)
        model.restore(save_dir=SAVE_SUBDIR)
    model.prepare_test_hash()
    performance = model.test_hash_metric(K_SET)
    model.delete()
    del model
    del dm_train
    del dm_val
    del dm_test

    write_pkl(performance, path=PKL_SUBDIR + 'evaluation.pkl')
    cwrite = CsvWriter2(1) 
    key_set = ['train_nmi', 'test_nmi', 'te_tr_suf', 'te_te_suf', 'te_te_precision_at_k', 'te_tr_precision_at_k']
    for key in key_set:
        cwrite.add_header(0, str(key))
        content = ''
        if 'at_k' in str(key): content = listformat(performance[key])
        else: content = performance[key]
        cwrite.add_content(0, content)
    cwrite.write(CSV_SUBDIR+'evaluation.csv')

def integrate_results():
    SAVE_DIR, LOG_DIR, CSV_DIR, PKL_DIR, BOARD_DIR, ASSET_DIR = subdirs5resultdir(RESULT_DIR, False)
    BOUNDARY = 10 
    FILE_KEY = 'evaluation.pkl'
    def get_value(path):
        content = read_pkl(path)
        value = np.sum(content['te_te_precision_at_k'])
        if np.mean(content['te_te_suf'])<BOUNDARY: return -1
        return value 
    max_value = -1
    max_path = None
    max_file = None
    for file in sorted(os.listdir(PKL_DIR)):
        PKL_SUBDIR = PKL_DIR+'{}/'.format(file)
        path = PKL_SUBDIR+'{}'.format(FILE_KEY)
        if os.path.exists(path):
            value = get_value(path)
            if max_value<value:
                max_value=value
                max_file = file
                max_path = path
    print(max_file)
    print(read_pkl(max_path))

if __name__ == '__main__':
    train_model()
    #integrate_results()

