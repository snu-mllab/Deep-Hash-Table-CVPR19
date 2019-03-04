import pickle
import numpy as np

def read_jpg(jpg_path, plt):
    '''
    Dependency : matplotlib.pyplot as plt
    Args:
        jpg_path - string
                   ends with jpg
        plt - plt object
    Return:
        numpy 3D image
    '''
    return plt.imread(jpg_path)

def read_pkl(path, encoding='ASCII'):
    '''read path(pkl) and return files
    Dependency : pickle
    Args:
        path - string
               ends with pkl
    Return:
        pickle content
    '''
    print("Pickle is read from %s"%path)
    with open(path, 'rb') as f: return pickle.load(f, encoding=encoding)

def read_txt(path):
    '''read txt files
    Args:
        path - string
               ends with txt
    Return:
        txt_content - list
            line by line
    '''
    print("Txt is read from %s"%path)

    txt_content = list()
    with open(path, 'r') as lines:
        for line in lines: txt_content.append(line)
    return txt_content

def read_npy(path):
    '''read npy files
    Args:
        path - string
               ends with npy
    Return:
        npy_content in path
    '''
    print("Npy is read from %s"%path)
    npy_content = np.load(path)
    return npy_content
