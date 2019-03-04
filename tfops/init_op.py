import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from tfops.info_op import get_uninit_vars

import tensorflow as tf

def rest_initializer(sess):
    print("Initialize uninitialized variables")
    sess.run(tf.variables_initializer(get_uninit_vars(sess)))

def full_initializer(sess):
    print("Initialize all variables")
    sess.run(tf.global_variables_initializer())

def local_initializer(sess, var_list, print_option=False):
    if print_option: print("Initialize specific variables")
    sess.run(tf.variables_initializer(var_list))

