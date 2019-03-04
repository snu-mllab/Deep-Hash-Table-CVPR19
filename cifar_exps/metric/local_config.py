import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from configs.parser import cifar_parser
from configs.path import EXPDIR

import numpy as np
import argparse

KEY = 'cifar_metric_npairs_64'
RESULT_DIR = EXPDIR+"{}/".format(KEY)

ID_STRUCTURE_DICT = {
        'cifar_metric_npairs_64' : ('param', 'dptype'),
        'cifar_metric_triplet_256' : ('param', 'dptype'),
        }

ID_STRUCTURE = ID_STRUCTURE_DICT[KEY]

def local_cifar_parser():
    parser = cifar_parser()
    parser.add_argument("--nsclass", default = 64, help="the number of selected class", type = int)
    parser.add_argument("--ltype", default = 'triplet', help="loss type", type = str)
    parser.add_argument("--dtype", default = 'stair', help="decay type", type = str)
    parser.add_argument("--dptype", default = 'a5', help="hash decay param type", type = str)
    
    if KEY in ['cifar_metric_npairs_64']:
        parser.add_argument("--m", default = 64, help="continous representation size", type = int)
        parser.add_argument("--param", default = 0.003, help="reg", type = float)
    elif KEY in ['cifar_metric_triplet_256']:
        parser.add_argument("--m", default = 256, help="continous representation size", type = int)
        parser.add_argument("--param", default = 0.3, help="margin alpha", type = float)
    return parser

