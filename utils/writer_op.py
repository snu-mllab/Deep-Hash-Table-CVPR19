import pickle
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

def append_dict_pkl(dict_, path):
    ''' 
    Args:
        dict_ - python dictionary
        path - string
            ends with pkl
            the content in path should be dict
    '''
    with open(path, 'rb') as f:
        print("Pickle is read from %s"%path)
        content=pickle.load(f)

    assert isinstance(dict_, dict) and isinstance(content, dict), 'dict_ and content should be type of dictionary' 

    for key in dict_.keys():
        content[key] = dict_[key]

    with open(path, 'wb') as f:
        print("Pickle is written on %s"%path)
        try:
            pickle.dump(content, f)
        except OverflowError:
            pickle.dump(content, f, protocol=4)

def create_dir(dirname):
    '''create directory named dirname
    Dependency : os
    Args:
        dirname - string
                  directory named
    '''
    if not os.path.exists(dirname):
        print("Creating %s"%dirname)
        os.makedirs(dirname)
    else:
        print("Already %s exists"%dirname)

def create_muldir(*args):
    for dirname in args:
        create_dir(dirname) 

def write_pkl(content, path):
    '''write content on path with path
    Dependency : pickle
    Args:
        content - object to be saved
        path - string
                ends with pkl
    '''
    with open(path, 'wb') as f:
        print("Pickle is written on %s"%path)
        try:
            pickle.dump(content, f)
        except OverflowError:
            pickle.dump(content, f, protocol=4)

def write_npy(content, path):
    '''write content on path with path
    Dependency : numpy as np
    Args:
        content - object to be saved
        path - string
                ends with npy
    '''
    print("Numpy is written on %s"%path)
    np.save(path, content)

class MatplotlibPdfManager:
    def __init__(self, path, plt):
        self.path = path
        print("Creating {}".format(self.path))
        self.pdf = PdfPages(self.path)
        self.plt = plt
        self.ncount = 0

    def generate_from(self):
        self.ncount+=1
        print("From here generating {} pages in {}".format(self.ncount, self.path))
        self.plt.close()

    def generate_to(self):
        print("To here generating {} pages in {}".format(self.ncount, self.path))
        self.pdf.savefig()

