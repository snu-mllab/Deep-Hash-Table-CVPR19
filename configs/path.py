import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))
from utils.writer_op import create_muldir, create_dir

#=============================PATH==========================================#
ROOT = '/data/unsynced_store/maestro/dht_github/' # User define
#=============================INITIAL DOWNLOAD==============================#
CIFAR100PATH = ROOT+'cifar-100-python/
#============================EXPRESULTS=====================================#
EXPDIR = ROOT+'exp_results/'
#=============================DATASET======================================#
CIFARPROCESSED = ROOT+'cifar_processed/'
#===========================================================================#
def subdirs5resultdir(result_dir, generate_option=False):
    save_dir = result_dir+'save/'
    log_dir = result_dir+'log/'
    csv_dir = result_dir+'csv/' 
    pkl_dir = result_dir+'pkl/' 
    board_dir = result_dir+'board/'
    asset_dir = result_dir+'asset/'
    if generate_option: create_muldir(save_dir, log_dir, csv_dir, pkl_dir, board_dir, asset_dir)

    return save_dir, log_dir, csv_dir, pkl_dir, board_dir, asset_dir

def dir2subdir(dir_path, file_id, generate_option=False):
    subdir_path = dir_path+'%s/'%file_id
    if generate_option: create_dir(subdir_path)
    return subdir_path

def muldir2mulsubdir(dir_pathes, file_id, generate_option=False):
    subdir_pathes = list()
    for dir_path in dir_pathes: subdir_pathes.append(dir2subdir(dir_path=dir_path, file_id=file_id, generate_option=generate_option))
    return subdir_pathes





