import tensorflow as tf
from ops import clip_ts

def cross_entropy_loss1(objective, logits):
    '''
    Reduce KL divergence value
    Args:
        objective - 2D tensor 
        logits - 2D tensor
    Return: 
        cost tensor
    '''
    objective/=tf.reduce_sum(objective, axis=1, keep_dims=True) 
    cost = tf.reduce_mean(tf.negative(tf.reduce_sum(tf.multiply(objective, tf.log(tf.nn.softmax(logits, dim=-1))), axis=-1)))
    return cost

def ranking_loss1(objective, logits, margin):
    '''
    Args:
        objective - 2D tensor 
        logits - 2D tensor
        margin - float
    Return: 
        ranking_loss -  tensor
    '''
    ranking_loss = tf.add(tf.subtract(logits, tf.reduce_sum(tf.multiply(objective, logits), axis=1, keep_dims=True)), margin)
    ranking_loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(ranking_loss, 0.0), axis=1))
    return ranking_loss

HASH_CROSS_ENTROPY_LOSS_DICT = {
        'h1' : cross_entropy_loss1
    }
    
HASH_RANKING_LOSS_DICT = {
        'h1' : ranking_loss1
    }

