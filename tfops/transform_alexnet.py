import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from configs.path import ROOT

from utils.writer_op import create_muldir
from utils.gpu_op import selectGpuById

from tfops.init_op import full_initializer
from tfops.info_op import vars_info_vl, get_uninit_vars, vars_info

import tensorflow as tf
import numpy as np

def get_shape(t):
    '''get the shape of tensor as list
    Args:
        t - tensor
    Return:
        list - shape of t
    '''
    return [str(v) for v in t.get_shape().as_list()]

def save(npy_path, save_dir, gpu_id=0):
    '''
    conv1
    kernel shape
    [11,11,3,96]
    biases shape
    [96]
    conv2
    kernel shape
    [5,5,48,256]
    biases shape
    [256]
    conv3
    kernel shape
    [3,3,256,384]
    biases shape
    [384]
    conv4
    kernel shape
    [3,3,192,384]
    biases shape
    [384]
    conv5
    kernel shape
    [3,3,192,256]
    biases shape
    [256]
    fc6
    fc6w shape
    [9216,4096]
    fc6b shape
    [4096]
    fc7
    fc7w shape
    [4096,4096]
    fc7b shape
    [4096]
    fc8
    fc8w shape
    [4096,1000]
    fc8b shape
    [1000]
    0.0166103
    '''
    create_muldir(save_dir)
    selectGpuById(gpu_id)

    net_data = dict(np.load(npy_path, encoding='bytes').item())
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(net_data['conv1'][0], name='weights')
        biases = tf.Variable(net_data['conv1'][1], name='biases')
    print('conv1')
    print("kernel shape\n[{}]".format(','.join(get_shape(kernel))))
    print("biases shape\n[{}]".format(','.join(get_shape(biases))))
    with tf.variable_scope('conv2') as scope:
        kernel = tf.Variable(net_data['conv2'][0], name='weights')
        biases = tf.Variable(net_data['conv2'][1], name='biases')
    print('conv2')
    print("kernel shape\n[{}]".format(','.join(get_shape(kernel))))
    print("biases shape\n[{}]".format(','.join(get_shape(biases))))
    with tf.variable_scope('conv3') as scope:
        kernel = tf.Variable(net_data['conv3'][0], name='weights')
        biases = tf.Variable(net_data['conv3'][1], name='biases')
    print('conv3')
    print("kernel shape\n[{}]".format(','.join(get_shape(kernel))))
    print("biases shape\n[{}]".format(','.join(get_shape(biases))))
    with tf.variable_scope('conv4') as scope:
        kernel = tf.Variable(net_data['conv4'][0], name='weights')
        biases = tf.Variable(net_data['conv4'][1], name='biases')
    print('conv4')
    print("kernel shape\n[{}]".format(','.join(get_shape(kernel))))
    print("biases shape\n[{}]".format(','.join(get_shape(biases))))
    with tf.variable_scope('conv5') as scope:
        kernel = tf.Variable(net_data['conv5'][0], name='weights')
        biases = tf.Variable(net_data['conv5'][1], name='biases')
    print('conv5')
    print("kernel shape\n[{}]".format(','.join(get_shape(kernel))))
    print("biases shape\n[{}]".format(','.join(get_shape(biases))))
    with tf.variable_scope('fc6'):
        fc6w = tf.Variable(net_data['fc6'][0], name='weights')
        fc6b = tf.Variable(net_data['fc6'][1], name='biases')
    print('fc6')
    print("fc6w shape\n[{}]".format(','.join(get_shape(fc6w))))
    print("fc6b shape\n[{}]".format(','.join(get_shape(fc6b))))

    with tf.variable_scope('fc7'):
        fc7w = tf.Variable(net_data['fc7'][0], name='weights')
        fc7b = tf.Variable(net_data['fc7'][1], name='biases')
    print('fc7')
    print("fc7w shape\n[{}]".format(','.join(get_shape(fc7w))))
    print("fc7b shape\n[{}]".format(','.join(get_shape(fc7b))))

    with tf.variable_scope('fc8'):
        fc8w = tf.Variable(net_data['fc8'][0], name='weights')
        fc8b = tf.Variable(net_data['fc8'][1], name='biases')
    print('fc8')
    print("fc8w shape\n[{}]".format(','.join(get_shape(fc8w))))
    print("fc8b shape\n[{}]".format(','.join(get_shape(fc8b))))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)

    full_initializer(sess)
    print(sess.run(fc8w)[0][0])
    saver=tf.train.Saver(max_to_keep = 5)
    saver.save(sess, os.path.join(save_dir, 'model'), global_step = 0)

def restore(save_dir, gpu_id=0):
    '''
    0.0166103    
    Collection name trainable_variables
	conv1/weights:0 : [11, 11, 3, 96]
	conv1/biases:0 : [96]
	conv2/weights:0 : [5, 5, 48, 256]
	conv2/biases:0 : [256]
	conv3/weights:0 : [3, 3, 256, 384]
	conv3/biases:0 : [384]
	conv4/weights:0 : [3, 3, 192, 384]
	conv4/biases:0 : [384]
	conv5/weights:0 : [3, 3, 192, 256]
	conv5/biases:0 : [256]
	fc6/weights:0 : [9216, 4096]
	fc6/biases:0 : [4096]
	fc7/weights:0 : [4096, 4096]
	fc7/biases:0 : [4096]
	fc8/weights:0 : [4096, 1000]
	fc8/biases:0 : [1000]
    []
    '''
    selectGpuById(gpu_id)
    with tf.variable_scope('conv1'):
        kernel = tf.get_variable(name="weights", shape = [11,11,3,96], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases", shape = [96] , initializer= tf.constant_initializer(0.001))
    with tf.variable_scope('conv2') :
        kernel = tf.get_variable(name="weights", shape = [5,5,48,256] , initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases", shape = [256] , initializer= tf.constant_initializer(0.001))
    with tf.variable_scope('conv3') :
        kernel = tf.get_variable(name="weights", shape = [3,3,256,384] , initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases", shape = [384] , initializer= tf.constant_initializer(0.001))
    with tf.variable_scope('conv4') :
        kernel = tf.get_variable(name="weights", shape = [3,3,192,384] , initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases", shape = [384] , initializer= tf.constant_initializer(0.001))
    with tf.variable_scope('conv5') :
        kernel = tf.get_variable(name="weights", shape = [3,3,192,256] , initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(name="biases", shape = [256], initializer= tf.constant_initializer(0.001))
    with tf.variable_scope('fc6'):
        fc6w = tf.get_variable(name="weights", shape = [9216,4096] , initializer =tf.random_normal_initializer(stddev=1e-2))
        fc6b = tf.get_variable(name="biases", shape = [4096] , initializer =tf.constant_initializer(0.001))
    with tf.variable_scope('fc7'):
        fc7w = tf.get_variable(name="weights", shape = [4096,4096] , initializer =tf.random_normal_initializer(stddev=1e-2))
        fc7b = tf.get_variable(name="biases", shape = [4096] , initializer =tf.constant_initializer(0.001))
    with tf.variable_scope('fc8'):
        fc8w = tf.get_variable(name="weights", shape = [4096,1000] , initializer =tf.random_normal_initializer(stddev=1e-2))
        fc8b = tf.get_variable(name="biases", shape = [1000] , initializer =tf.constant_initializer(0.001))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(save_dir)
    saver.restore(sess, checkpoint)
    print(sess.run(fc8w)[0][0])
    print(vars_info("trainable_variables"))
    print(get_uninit_vars(sess))

if __name__=='__main__':
    npy_path = ROOT+'pretrain/reference_pretrain.npy'
    save_dir = ROOT+'pretrain/alexnet_save/'
    #save(npy_path=npy_path, save_dir=save_dir, gpu_id=0)
    restore(save_dir=save_dir, gpu_id=0)





