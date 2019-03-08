import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.np_op import np_center_crop_4d

from tensorflow.python.framework import dtypes, ops, sparse_tensor, tensor_shape
from tensorflow.python.ops import array_ops, control_flow_ops, logging_ops, math_ops,\
                                  nn, script_ops, sparse_ops

import tensorflow as tf
import numpy as np

def apply_tf_op(inputs, session, input_gate, output_gate, batch_size, dim=4, train_gate=None, crop_size=None):
    '''
    Requires the graph to be built alreadly
    Dependency:
        import tensorflow as tf
        import numpy as np
    Args:
        inputs - 2-D vector [ndata, nfeature]
                 4-D image [ndata, height, width, nchannel]
        session - session of tf to run
        input_gate - placeholder for tf operation
        output_gate - output tensor
        batch_size - int
        dim - int
        train_gate - determine whether train or not
    Return:
        outputs - N-D image [ndata]
    '''
    if dim == 4:
        ndata, height, width, nchannel = inputs.shape
        if ndata%batch_size!=0:
            inputs = np.concatenate([inputs, np.zeros([batch_size-ndata%batch_size, height, width, nchannel])], axis=0) 
    else:
        ndata, nfeature = inputs.shape
        if ndata%batch_size!=0:
            inputs = np.concatenate([inputs, np.zeros([batch_size-ndata%batch_size, nfeature])], axis=0) 

    nbatch = len(inputs)//batch_size

    outputs = list()
    for b in range(nbatch):
        if crop_size is None:
            feed_dict = {input_gate : inputs[b*batch_size:(b+1)*batch_size]}
        else:
            feed_dict = {input_gate : np_center_crop_4d(inputs[b*batch_size:(b+1)*batch_size], crop_size)}
        if train_gate is not None: feed_dict[train_gate] = False 
        outputs.append(session.run(output_gate, feed_dict=feed_dict))
    outputs = np.concatenate(outputs, axis=0)
    outputs = outputs[:ndata]
    return outputs

def pairwise_argsort_euclid_efficient(input1, input2, session, batch_size): 
    '''
    Args:
        input1 - Numpy 2D array [ndata1, nfeature]
        input2 - Numpy 2D array [ndata2, nfeature]
    '''
    assert input1.shape[1]==input2.shape[1], "input1, input2 should have same feature"

    ndata1, nfeature = input1.shape
    ndata2, _ = input2.shape

    input1_s = tf.placeholder(tf.float32, shape=[batch_size, nfeature])
    input2_t = tf.convert_to_tensor(input2, dtype=tf.float32)
    
    p_dist = math_ops.add(
    	math_ops.reduce_sum(math_ops.square(input1_s), axis=[1], keep_dims=True),
        math_ops.reduce_sum(
                math_ops.square(array_ops.transpose(input2_t)), axis=[0], keep_dims=True))-\
                2.0 * math_ops.matmul(input1_s, array_ops.transpose(input2_t))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    p_dist = math_ops.maximum(p_dist, 0.0)
    error_mask = math_ops.less_equal(p_dist, 0.0)
    p_dist = math_ops.multiply(p_dist, math_ops.to_float(math_ops.logical_not(error_mask)))
    argsort= tf.nn.top_k(-p_dist, k=ndata2)[1]

    return apply_tf_op(inputs=input1, session=session, input_gate=input1_s, output_gate=argsort, batch_size=batch_size, dim=2)


def pairwise_distance_euclid_efficient(input1, input2, session, batch_size): 
    '''
    Args:
        input1 - Numpy 2D array [ndata1, nfeature]
        input2 - Numpy 2D array [ndata2, nfeature]
    '''
    assert input1.shape[1]==input2.shape[1], "input1, input2 should have same feature"

    ndata1, nfeature = input1.shape
    ndata2, _ = input2.shape

    input1_s = tf.placeholder(tf.float32, shape=[batch_size, nfeature])
    
    input2_t = tf.convert_to_tensor(input2, dtype=tf.float32)
    
    p_dist = math_ops.add(
    	math_ops.reduce_sum(math_ops.square(input1_s), axis=[1], keep_dims=True),
        math_ops.reduce_sum(
                math_ops.square(array_ops.transpose(input2_t)), axis=[0], keep_dims=True))-\
                2.0 * math_ops.matmul(input1_s, array_ops.transpose(input2_t))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    p_dist = math_ops.maximum(p_dist, 0.0)
    error_mask = math_ops.less_equal(p_dist, 0.0)
    p_dist = math_ops.multiply(p_dist, math_ops.to_float(math_ops.logical_not(error_mask)))

    return apply_tf_op(inputs=input1, session=session, input_gate=input1_s, output_gate=p_dist, batch_size=batch_size, dim=2)

def get_recall_at_1_efficient(data, label, input1_tensor, input2_tensor, idx_tensor, session): 
    '''
    args:
        data - numpy 2d array [ndata, nfeature]
        label - numpy 1d array [ndata]
        input1_tensor - placeholder [batch_size, nfeature]
        input2_tensor - placeholder [ndata, nfeature]
        idx_tensor - tensor [ndata, 2]
    return:
        recall
    '''
    batch_size = input1_tensor.get_shape().as_list()[0]
    ndata, nfeature = data.shape

    if ndata%batch_size!=0:
        inputs = np.concatenate([data, np.zeros([batch_size-ndata%batch_size, nfeature])], axis=0) 
    nbatch = len(inputs)//batch_size

    outputs = np.zeros([len(inputs), 2])
    for b in range(nbatch):
        feed_dict = {
                    input1_tensor : inputs[b*batch_size:(b+1)*batch_size],\
                    input2_tensor : data
                    }
        outputs[b*batch_size:(b+1)*batch_size]=session.run(idx_tensor, feed_dict=feed_dict)
    outputs = outputs[:ndata] # [ndata, 2]

    nsuccess = 0
    for idx1 in range(ndata):
        for idx2 in outputs[idx1]:
            if int(idx1)==int(idx2):
                continue
            if label[int(idx1)]==label[int(idx2)]: 
                nsuccess+=1
                break
    return nsuccess/ndata

def get_recall_at_1_efficient_selective(embed_data, idx_data, label, embed_tensor1, embed_tensor2, idx_tensor1, idx_tensor2, output_tensor, session): 
    '''
    args:
        embed_data - numpy 2d array [ndata, nfeature]
        idx_data - numpy 2d array [ndata, 1]
        label - numpy 1d array [ndata]
        embed_tensor1 - placeholder [batch_size, nfeature]
        embed_tensor2 - placeholder [ndata, nfeature]
        idx_tensor1 - placeholder [batch_size, 1]
        idx_tensor2 - placeholder [ndata, 1]
        output_tensor - tensor [ndata, 2]
    return:
        recall
    '''
    batch_size = embed_tensor1.get_shape().as_list()[0]
    ndata, nfeature = embed_data.shape
    assert idx_data.shape==(ndata, 1), "wrong idx_data shape"

    if ndata%batch_size!=0:
        embed_inputs = np.concatenate([embed_data, np.zeros([batch_size-ndata%batch_size, nfeature])], axis=0) 
        idx_inputs = np.concatenate([idx_data, np.zeros([batch_size-ndata%batch_size, 1])], axis=0) 
    nbatch = len(embed_inputs)//batch_size

    outputs = np.zeros([len(embed_inputs), 2])
    for b in range(nbatch):
        feed_dict = {
                    embed_tensor1 : embed_inputs[b*batch_size:(b+1)*batch_size],\
                    embed_tensor2 : embed_data,\
                    idx_tensor1 : idx_inputs[b*batch_size:(b+1)*batch_size],\
                    idx_tensor2 : idx_data
                    }
        outputs[b*batch_size:(b+1)*batch_size]=session.run(output_tensor, feed_dict=feed_dict)
    outputs = outputs[:ndata] # [ndata, 2]

    nsuccess = 0
    for idx1 in range(ndata):
        for idx2 in outputs[idx1]:
            if int(idx1)==int(idx2): continue
            if label[int(idx1)]==label[int(idx2)]: 
                nsuccess+=1
                break
    return nsuccess/ndata

class HashDistanceManager:
    def __init__(self, batch_size, ndata, nfeature): 
        self.batch_size = batch_size
        self.ndata = ndata
        self.nfeature = nfeature
        self.tensor1 = tf.placeholder(tf.int64, shape=[self.batch_size, self.nfeature])
        self.tensor2 = tf.placeholder(tf.int64, shape=[self.ndata, self.nfeature])

        self.p_dist = tf.reduce_sum(tf.cast(tf.logical_not(tf.equal(tf.expand_dims(self.tensor1, axis=1), tf.expand_dims(self.tensor2, axis=0))), dtype=tf.float32), axis=-1)
        # [batch_size, ndata]
        self.p_argsort = tf.nn.top_k(-self.p_dist, k=self.ndata)[1]

    def calculate(self, data, session):
        ndata, nfeature = data.shape
        assert ndata==self.ndata, "Wrong data input"

        if ndata%self.batch_size!=0:
            inputs = np.concatenate([data, np.zeros([self.batch_size-ndata%self.batch_size, self.nfeature])], axis=0) 
        nbatch = len(inputs)//self.batch_size

        outputs1 = np.zeros([len(inputs), self.ndata])
        outputs2 = np.zeros([len(inputs), self.ndata], dtype=np.int32)
        feed_dict = dict()
        feed_dict[self.tensor2]=data
        for b in range(nbatch):
            feed_dict[self.tensor1] = inputs[b*self.batch_size:(b+1)*self.batch_size]
            outputs1[b*self.batch_size:(b+1)*self.batch_size]=session.run(self.p_dist, feed_dict=feed_dict)
            outputs2[b*self.batch_size:(b+1)*self.batch_size]=session.run(self.p_argsort, feed_dict=feed_dict)
        outputs1 = outputs1[:ndata] # [ndata, ndata]
        outputs2 = outputs2[:ndata] # [ndata, ndata]

        hashdist_matrix = outputs1
        hashargsort_matrix = outputs2

        return hashdist_matrix, hashargsort_matrix

    def calculateV2(self, data1, data2, session):
        ndata1, nfeature = data1.shape
        ndata2, nfeature = data2.shape
        assert ndata2==self.ndata, "Wrong data input"

        if ndata1%self.batch_size!=0:
            inputs = np.concatenate([data1, np.zeros([self.batch_size-ndata1%self.batch_size, self.nfeature])], axis=0) 
        nbatch = len(inputs)//self.batch_size

        outputs1 = np.zeros([len(inputs), self.ndata])
        outputs2 = np.zeros([len(inputs), self.ndata], dtype=np.int32)
        feed_dict = dict()
        feed_dict[self.tensor2]=data2
        for b in range(nbatch):
            feed_dict[self.tensor1] = inputs[b*self.batch_size:(b+1)*self.batch_size]
            outputs1[b*self.batch_size:(b+1)*self.batch_size]=session.run(self.p_dist, feed_dict=feed_dict)
            outputs2[b*self.batch_size:(b+1)*self.batch_size]=session.run(self.p_argsort, feed_dict=feed_dict)
        outputs1 = outputs1[:ndata1] # [ndata1, ndata2]
        outputs2 = outputs2[:ndata1] # [ndata1, ndata2]

        hashdist_matrix = outputs1
        hashargsort_matrix = outputs2

        return hashdist_matrix, hashargsort_matrix

if __name__=='__main__':
    from dist import pairwise_distance_euclid_v2
    
    sess= tf.Session()
    array1 = np.random.uniform(size=[5, 10])
    array2 = np.random.uniform(size=[5, 10])
    print(pairwise_distance_euclid_efficient(array1, array1, session=sess, batch_size=100))
    print(pairwise_argsort_euclid_efficient(array1, array1, session=sess, batch_size=100))
    


