import tensorflow as tf
import numpy as np

def pool2d(x, per_op, op_name):
    assert op_name in ['AVG', 'MAX'], "op_name arg should be 'AVG' or 'MAX'"

    if op_name == 'AVG':
        return tf.nn.pool(x, window_shape = per_op, pooling_type = 'AVG', strides = per_op, padding='VALID')
    else:
        return tf.nn.pool(x, window_shape = per_op, pooling_type = 'MAX', strides = per_op, padding='VALID')

def max_pool2d(x, per_max):
    '''
    max_pool2d, wrapper of pool2d
    Args:
        x - 4d tensor
            'NHWC' batch, height, width, channel
        per_max - list with two ints
    '''
    return pool2d(x, per_op=per_max, op_name='MAX')

def avg_pool2d(x, per_avg):
    '''
    avg_pool2d wrapper of pool2d
    Args:
        x - 4d tensor
            'NHWC' batch, height, width, channel
        per_avg - list with two ints
    '''
    return pool2d(x, per_op=per_avg, op_name='AVG')

def conv2d(input_, filter_shape, strides = [1,1,1,1], padding = False, activation = None, batch_norm = False, istrain = False, scope = None):
    '''
    Args:
        input_ - 4D tensor
            Normally NHWC format
        filter_shape - 1D array 4 elements
            [height, width, inchannel, outchannel]
        strides - 1D array 4 elements
            default to be [1,1,1,1]
        padding - bool
            Deteremines whether add padding or not
            True => add padding 'SAME'
            Fale => no padding  'VALID'
        activation - activation function
            default to be None
        batch_norm - bool
            default to be False
            used to add batch-normalization
        istrain - bool
            indicate the model whether train or not
        scope - string
            default to be None
    Return:
        4D tensor
        activation(batch(conv(input_)))
    '''
    with tf.variable_scope(scope or "conv"):
        if padding:
            padding = 'SAME'
        else:
            padding = 'VALID'
        w = tf.get_variable(name="weight", shape = filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)) 
        conv = tf.nn.conv2d(input_, w, strides=strides, padding=padding)
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay = 0.99, is_training=istrain, scope='batch_norm')
            if activation is None:
                return norm
            return activation(norm)
        else:
            b = tf.get_variable(name="bias", shape = filter_shape[-1], initializer=tf.constant_initializer(0.001))
            if activation is None:
                return conv + b
            return activation(conv + b)

def fc_layer(input_, output_size, activation = None, batch_norm = False, istrain = False, scope = None):
    '''
    fully convlolution layer
    Args :
        input_  - 2D tensor
            general shape : [batch, input_size]
        output_size - int
            shape of output 2D tensor
        activation - activation function
            defaults to be None
        batch_norm - bool
            defaults to be False
            if batch_norm to apply batch_normalization
        istrain - bool
            defaults to be False
            indicator for phase train or not
        scope - string
            defaults to be None then scope becomes "fc"
    '''
    with tf.variable_scope(scope or "fc"):
        w = tf.get_variable(name="weight", shape = [get_shape(input_)[1], output_size], initializer=tf.contrib.layers.xavier_initializer())
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(tf.matmul(input_, w) , center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            if activation is None:
                return norm
            return activation(norm)
        else:
            b = tf.get_variable(name="bias", shape = [output_size], initializer=tf.constant_initializer(0.0))
            if activation is None:
                return tf.nn.xw_plus_b(input_, w, b)
            return activation(tf.nn.xw_plus_b(input_, w, b))

