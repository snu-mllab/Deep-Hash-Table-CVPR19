import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.gpu_op import selectGpuById
from tfops.hash_dist import pairwise_distance_w_obj1

import tensorflow as tf

def pairwise_distance_w_obj1_test(gpuid=0):
    selectGpuById(gpuid)
    sess = tf.Session()
    feature = tf.constant([[1,2,3], [-1,-1,0], [1,0,2], [1,2,1]], dtype=tf.float32) 
    objective = tf.constant([[0,0,1], [0,0,1], [0,0,1], [0,1,0]], dtype=tf.float32) 
    print("feature : \n", sess.run(feature))
    print("objective : \n", sess.run(objective))
    print("pairwisedist : \n", sess.run(pairwise_distance_w_obj1(feature, objective)))

if __name__ == '__main__':
    pairwise_distance_w_obj1_test()
