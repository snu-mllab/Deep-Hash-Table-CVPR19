import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from configs.parser import cifar_parser
from configs.path import EXPDIR

import numpy as np
import argparse

KEY = 'cifar_ours_triplet_64_2_1'
RESULT_DIR = EXPDIR+"{}/".format(KEY)

ID_STRUCTURE_DICT = {
        'cifar_ours_npairs_32_2_1' : ('param', 'plamb2', 'dptype'),
        'cifar_ours_npairs_32_2_2' : ('param', 'plamb2', 'dptype'),
        'cifar_ours_npairs_32_2_3' : ('param', 'plamb2', 'dptype'),
        'cifar_ours_npairs_32_2_4' : ('param', 'plamb2', 'dptype'),
        'cifar_ours_triplet_64_2_1' : ('param', 'plamb2', 'dptype'),
        'cifar_ours_triplet_64_2_2' : ('param', 'plamb2', 'dptype'),
        'cifar_ours_triplet_64_2_3' : ('param', 'plamb2', 'dptype'),
        'cifar_ours_triplet_64_2_4' : ('param', 'plamb2', 'dptype'),
        }

ID_STRUCTURE = ID_STRUCTURE_DICT[KEY]

def local_cifar_parser():
    parser = cifar_parser()
    parser.add_argument("--nsclass", default = 64, help="the number of selected class", type = int)
    parser.add_argument("--label", default = 'dynamic', help="how to add label static or dynamic", type = str) # dynamic => label remapping, static => no label remapping
    parser.add_argument("--plamb1", default = 100.0, help="lambda for pairwise cost", type = float)
    parser.add_argument("--dtype", default = 'stair', help="decay type", type = str)
    parser.add_argument("--dptype", default = 'a5', help="hash decay param type", type = str)
    
    if KEY in ['cifar_ours_npairs_32_2_1']:
        parser.add_argument("--d", default = 32, help="bucket d", type = int)
        parser.add_argument("--k", default = 2, help="number of hierachical", type = int)
        parser.add_argument("--sk", default = 1, help="sparse k", type = int)
        parser.add_argument("--meta", default=EXPDIR+'cifar_metric_npairs_64/meta/meta.pkl', type=str)
        parser.add_argument("--save", default=EXPDIR+'cifar_metric_npairs_64/bestsave/', type=str)
        parser.add_argument("--ltype", default = 'npair', help="loss type", type = str)
        parser.add_argument("--param", default = 0.03, help="hash reg", type = float)
        parser.add_argument("--plamb2", default = 0.07, help="lambda for pairwise cost", type = float)
    elif KEY in ['cifar_ours_npairs_32_2_2']:
        parser.add_argument("--d", default = 32, help="bucket d", type = int)
        parser.add_argument("--k", default = 2, help="number of hierachical", type = int)
        parser.add_argument("--sk", default = 2, help="sparse k", type = int)
        parser.add_argument("--meta", default=EXPDIR+'cifar_metric_npairs_64/meta/meta.pkl', type=str)
        parser.add_argument("--save", default=EXPDIR+'cifar_metric_npairs_64/bestsave/', type=str)
        parser.add_argument("--ltype", default = 'npair', help="loss type", type = str)
        parser.add_argument("--param", default = 0.01, help="hash reg", type = float)
        parser.add_argument("--plamb2", default = 0.1, help="lambda for pairwise cost", type = float)
    elif KEY in ['cifar_ours_npairs_32_2_3']:
        parser.add_argument("--d", default = 32, help="bucket d", type = int)
        parser.add_argument("--k", default = 2, help="number of hierachical", type = int)
        parser.add_argument("--sk", default = 3, help="sparse k", type = int)
        parser.add_argument("--meta", default=EXPDIR+'cifar_metric_npairs_64/meta/meta.pkl', type=str)
        parser.add_argument("--save", default=EXPDIR+'cifar_metric_npairs_64/bestsave/', type=str)
        parser.add_argument("--ltype", default = 'npair', help="loss type", type = str)
        parser.add_argument("--param", default = 0.003, help="hash reg", type = float)
        parser.add_argument("--plamb2", default = 0.1, help="lambda for pairwise cost", type = float)
    elif KEY in ['cifar_ours_npairs_32_2_4']:
        parser.add_argument("--d", default = 32, help="bucket d", type = int)
        parser.add_argument("--k", default = 2, help="number of hierachical", type = int)
        parser.add_argument("--sk", default = 4, help="sparse k", type = int)
        parser.add_argument("--meta", default=EXPDIR+'cifar_metric_npairs_64/meta/meta.pkl', type=str)
        parser.add_argument("--save", default=EXPDIR+'cifar_metric_npairs_64/bestsave/', type=str)
        parser.add_argument("--ltype", default = 'npair', help="loss type", type = str)
        parser.add_argument("--param", default = 0.001, help="hash reg", type = float)
        parser.add_argument("--plamb2", default = 0.1, help="lambda for pairwise cost", type = float)
    elif KEY in ['cifar_ours_triplet_64_2_1']:
        parser.add_argument("--d", default = 64, help="bucket d", type = int)
        parser.add_argument("--k", default = 2, help="number of hierachical", type = int)
        parser.add_argument("--sk", default = 1, help="sparse k", type = int)
        parser.add_argument("--meta", default=EXPDIR+'cifar_metric_triplet_256/meta/meta.pkl', type=str)
        parser.add_argument("--save", default=EXPDIR+'cifar_metric_triplet_256/bestsave/', type=str)
        parser.add_argument("--ltype", default = 'triplet', help="loss type", type = str)
        parser.add_argument("--param", default = 0.5, help="hash margin alpha", type = float)
        parser.add_argument("--plamb2", default = 1.0, help="lambda for pairwise cost", type = float)
    elif KEY in ['cifar_ours_triplet_64_2_2']:
        parser.add_argument("--d", default = 64, help="bucket d", type = int)
        parser.add_argument("--k", default = 2, help="number of hierachical", type = int)
        parser.add_argument("--sk", default = 2, help="sparse k", type = int)
        parser.add_argument("--meta", default=EXPDIR+'cifar_metric_triplet_256/meta/meta.pkl', type=str)
        parser.add_argument("--save", default=EXPDIR+'cifar_metric_triplet_256/bestsave/', type=str)
        parser.add_argument("--ltype", default = 'triplet', help="loss type", type = str)
        parser.add_argument("--param", default = 0.3, help="hash margin alpha", type = float)
        parser.add_argument("--plamb2", default = 1.0, help="lambda for pairwise cost", type = float)
    elif KEY in ['cifar_ours_triplet_64_2_3']:
        parser.add_argument("--d", default = 64, help="bucket d", type = int)
        parser.add_argument("--k", default = 2, help="number of hierachical", type = int)
        parser.add_argument("--sk", default = 3, help="sparse k", type = int)
        parser.add_argument("--meta", default=EXPDIR+'cifar_metric_triplet_256/meta/meta.pkl', type=str)
        parser.add_argument("--save", default=EXPDIR+'cifar_metric_triplet_256/bestsave/', type=str)
        parser.add_argument("--ltype", default = 'triplet', help="loss type", type = str)
        parser.add_argument("--param", default = 0.5, help="hash margin alpha", type = float)
        parser.add_argument("--plamb2", default = 1.0, help="lambda for pairwise cost", type = float)
    elif KEY in ['cifar_ours_triplet_64_2_4']:
        parser.add_argument("--d", default = 64, help="bucket d", type = int)
        parser.add_argument("--k", default = 2, help="number of hierachical", type = int)
        parser.add_argument("--sk", default = 4, help="sparse k", type = int)
        parser.add_argument("--meta", default=EXPDIR+'cifar_metric_triplet_256/meta/meta.pkl', type=str)
        parser.add_argument("--save", default=EXPDIR+'cifar_metric_triplet_256/bestsave/', type=str)
        parser.add_argument("--ltype", default = 'triplet', help="loss type", type = str)
        parser.add_argument("--param", default = 0.5, help="hash margin alpha", type = float)
        parser.add_argument("--plamb2", default = 1.0, help="lambda for pairwise cost", type = float)
    return parser

