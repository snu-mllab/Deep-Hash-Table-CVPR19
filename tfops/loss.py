import tensorflow as tf
import numpy as np

def sigmoid_cross_entropy(labels, logits):
    '''
    Args:
        labels - N-D tensor
        logits - N-D tensor
    '''
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

def softmax_cross_entropy(labels, logits):
    '''
    Args:
        labels - (N-1)-D tensor int32 or int64
        logits - N-D tensor float32 
    '''
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

