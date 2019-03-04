import sys

sys.path.append('../utils')

from ops import dense_sum_list, get_shape
from np_op import convert2list
import tensorflow as tf
import numpy as np
import os

def test1(id_=0):
    ''' dense_sum_list test
    Results : 
        =================Round1===================
        const : [4, 2] 
        [[  1.   4.]
         [  3.   4.]
         [  5.   6.]
         [  1.  10.]]
        split :
        [array([[ 1.,  4.],
               [ 3.,  4.]], dtype=float32), array([[  5.,   6.],
               [  1.,  10.]], dtype=float32)]
        dense_sum : (2, 4)
        [[  6.   7.   9.  10.]
         [  4.  13.   5.  14.]]
        =================Round2===================
        const : [3, 2]
        [[    1.    10.]
         [    2.   100.]
         [    3.  1000.]]
        split :
        [array([[  1.,  10.]], dtype=float32), array([[   2.,  100.]], dtype=float32), array([[    3.,  1000.]], dtype=float32)]
        dense_sum : (1, 8)
        [[    6.  1003.   104.  1101.    15.  1012.   113.  1110.]]
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(id_)
    sess = tf.Session()

    print("=================Round1===================")
    const = np.array([[1,4],[3,4],[5,6],[1,10]])
    const = tf.constant(const, dtype=tf.float32)
    print("const : {} \n{}".format(get_shape(const), sess.run(const)))
    split = tf.split(const, num_or_size_splits=2, axis=0)
    print("split :\n{}".format(sess.run(split)))
    ds = dense_sum_list(split)
    ds_run = sess.run(ds)
    print("dense_sum : {}\n{}".format(ds_run.shape, ds_run))
    print("=================Round2===================")
    const = np.array([[1,10],[2,100],[3,1000]])
    const = tf.constant(const, dtype=tf.float32)
    print("const : {}\n{}".format(get_shape(const), sess.run(const)))
    split = tf.split(const, num_or_size_splits=3, axis=0)
    print("split :\n{}".format(sess.run(split)))
    ds = dense_sum_list(split)
    ds_run = sess.run(ds)
    print("dense_sum : {}\n{}".format(ds_run.shape, ds_run))


def test2(id_=0):
    ''' test for
            cifar_exp/exp_9/deepmetric.py
            utils/eval_op/HashTree
    Results -

    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(id_)

    nbatch = 2
    k = 3 
    d = 4
    sparsity = 3 

    sess = tf.Session()
    const = np.random.random([nbatch*k, d])
    const = tf.constant(const, dtype=tf.float32)
    print("const : {} \n{}".format(get_shape(const), sess.run(const)))
    split = tf.split(const, num_or_size_splits=k, axis=0) # k*[batch_size, d]
    print("split :\n{}".format(sess.run(split)))
    tree_idx_set = [tf.nn.top_k(v, k=d)[1] for v in split]# k*[batch_size, d]
    print("tree_idx_set : \n{}".format(sess.run(tree_idx_set)))
    tree_idx = tf.transpose(tf.stack(tree_idx_set, axis=0), [1, 0, 2]) # [batch_size, k, d]
    
    print("tree_idx : \n{}".format(sess.run(tree_idx)))

    idx_convert_np = list()
    tmp = 1
    for i in range(k):
        idx_convert_np.append(tmp)
        tmp*=d
    idx_convert_np = np.array(idx_convert_np)[::-1] # [d**(k-1),...,1]
    idx_convert = tf.constant(idx_convert_np, dtype=tf.int32) # tensor [k]
    print("idx_convert :{} \n{}".format(get_shape(idx_convert), sess.run(idx_convert)))
    max_idx = tf.reduce_sum(tf.multiply(tree_idx[:,:,0], idx_convert), axis=-1) # [batch_size]
    print("max_idx :\n{}".format(sess.run(max_idx)))

    max_k_idx = tf.add(tf.reduce_sum(tf.multiply(tree_idx[:,:-1,0], idx_convert[:-1]), axis=-1, keep_dims=True), tree_idx[:,-1,:sparsity]) # [batch_size]
    print("max_k_idx :\n{}".format(sess.run(max_k_idx)))

    tree_idx = sess.run(tree_idx)
    for b_idx in range(nbatch): 
        for idx in range(d**k):
            idx_list = convert2list(n=idx, base=d, fill=k) # [k]
            st_idx= np.sum(np.multiply(np.array([tree_idx[b_idx][v][idx_list[v]] for v in range(k)]), idx_convert_np))
            print("b_idx, idx, st_idx : {}, {}, {}".format(b_idx, idx, st_idx))

def test3():
    const = np.array([[1,2,3],[30,20,10],[100,300,200],[3000,1000,2000]])
    const = tf.constant(const, dtype=tf.float32)
    print(get_shape(const)==[4,3])

def test4(id_=0):
    ''' test for
            icml_imgnet/expf5npair/deepmetric.py
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(id_)

    nbatch = 2
    sk = 3 
    nhash = 3
    d = 5

    sess = tf.Session()
    anc_embed_k_hash1 = tf.constant(np.random.random([nbatch, nhash]), dtype=tf.float32)
    print("anc_embed_k_hash1 : {} \n{}".format(get_shape(anc_embed_k_hash1), sess.run(anc_embed_k_hash1)))
    anc_embed_k_hash2 = tf.constant(np.random.random([nbatch, d]), dtype=tf.float32)
    print("anc_embed_k_hash2 : {} \n{}".format(get_shape(anc_embed_k_hash2), sess.run(anc_embed_k_hash2)))

    idx_array = tf.reshape(
                        tf.add(
                            tf.expand_dims(tf.multiply(tf.nn.top_k(anc_embed_k_hash1, k=nhash)[1], d), axis=2),
                            tf.expand_dims(tf.nn.top_k(anc_embed_k_hash2, k=d)[1], axis=1)),
                        [-1, nhash*d])
    # [nbatch, nhash, 1],  [nbatch, 1, d] => [nbatch//2, nhash, d] => [nbatch//2, nhash*d]
    max_k_idx = idx_array[:, :sk] # [batch_size, sk]

    print("max_k_idx :\n{}".format(sess.run(max_k_idx)))
    print("idx_array :\n{}".format(sess.run(idx_array)))

if __name__=='__main__':
    test4(0)
