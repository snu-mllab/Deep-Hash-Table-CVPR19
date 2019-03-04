import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from utils.general_class import ModelPlugin
from utils.evaluation import evaluate_metric_te_tr
from utils.sklearn_op import KMeansClustering
from utils.logger_op import LoggerManager
from utils.gpu_op import selectGpuById
from utils.tqdm_op import tqdm_range
from utils.np_op import activate_k_2D

from tfops.transform_op import apply_tf_op, pairwise_distance_euclid_efficient, get_recall_at_1_efficient
from tfops.summary_op import SummaryWriter
from tfops.train_op import get_multi_train_op
from tfops.info_op import vars_info_vl, get_init_vars, get_uninit_vars
from tfops.lr_op import DECAY_DICT, DECAY_PARAMS_DICT
from tfops.dist import npairs_loss, triplet_semihard_loss, pairwise_distance_euclid
from tfops.nets import conv1_32

from tqdm import tqdm
import tensorflow as tf
slim = tf.contrib.slim

from tensorflow.python.ops import array_ops, math_ops

import numpy as np
import glob
import os

class Model(ModelPlugin):
    def __init__(self, train_dataset, val_dataset, test_dataset, logfilepath, args):
        super().__init__(train_dataset, val_dataset, test_dataset, logfilepath, args)

    def build(self):
        self.logger.info("Model building starts")
        tf.reset_default_graph()
        if self.args.ltype == 'npair':
            self.anc_img = tf.placeholder(tf.float32, shape = [self.args.nbatch//2, self.height, self.width, self.nchannel])
            self.pos_img = tf.placeholder(tf.float32, shape = [self.args.nbatch//2, self.height, self.width, self.nchannel])
            self.istrain = tf.placeholder(tf.bool, shape= [])
            self.label = tf.placeholder(tf.int32, shape = [self.args.nbatch//2])
        else: # triplet
            self.img = tf.placeholder(tf.float32, shape = [self.args.nbatch, self.height, self.width, self.nchannel])
            self.istrain = tf.placeholder(tf.bool, shape= [])
            self.label = tf.placeholder(tf.int32, shape = [self.args.nbatch])
        
        self.generate_sess()
        self.conv_net = conv1_32
        
        if self.args.ltype == 'npair':
            self.anc_last = tf.nn.relu(self.conv_net(self.anc_img, is_training=self.istrain, reuse=False))
            self.pos_last = tf.nn.relu(self.conv_net(self.pos_img, is_training=self.istrain, reuse=True))
        else:#triplet
            self.last = tf.nn.relu(self.conv_net(self.img, is_training=self.istrain, reuse=False))

        with slim.arg_scope([slim.fully_connected], activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0005), biases_initializer=tf.zeros_initializer()):
            if self.args.ltype == 'npair':
                with tf.variable_scope('Embed', reuse=False): self.anc_embed = slim.fully_connected(self.anc_last, self.args.m, scope="fc1")
                with tf.variable_scope('Embed', reuse=True): self.pos_embed = slim.fully_connected(self.pos_last, self.args.m, scope="fc1")
                self.loss = npairs_loss(labels=self.label, embeddings_anchor=self.anc_embed, embeddings_positive=self.pos_embed, reg_lambda=self.args.param)
            else:#triplet
                with tf.variable_scope('Embed', reuse=False): self.embed = slim.fully_connected(self.last, self.args.m, scope = "fc1")
                self.embed_l2_norm = tf.nn.l2_normalize(self.embed, dim=-1) # embedding with l2 normalization
                def pairwise_distance_c(embeddings): return pairwise_distance_euclid(embeddings, squared=True)
                self.loss = triplet_semihard_loss(labels=self.label, embeddings=self.embed_l2_norm, pairwise_distance=pairwise_distance_c, margin=self.args.param)
        self.loss += tf.losses.get_regularization_loss()

        init_vars=get_init_vars(self.sess)
        self.logger.info("Variables loaded from pretrained network\n{}".format(vars_info_vl(init_vars)))
        uninit_vars=get_uninit_vars(self.sess)
        self.logger.info("Uninitialized variables\n{}".format(vars_info_vl(uninit_vars)))
        self.logger.info("Model building ends")
    
    def set_up_train(self):
        self.logger.info("Model setting up train starts")

        decay_func = DECAY_DICT[self.args.dtype]
        self.lr, update_step_op = decay_func(**DECAY_PARAMS_DICT[self.args.dtype][self.args.nbatch][self.args.dptype])

        update_ops = tf.get_collection("update_ops")

        var_slow_list, var_fast_list = list(), list()
        for var in tf.trainable_variables():
            if 'Embed' in var.name: var_fast_list.append(var)
            else: var_slow_list.append(var)

        with tf.control_dependencies(update_ops+[update_step_op]): self.train_op = get_multi_train_op(tf.train.AdamOptimizer, self.loss, [0.1*self.lr, self.lr], [var_slow_list, var_fast_list])

        self.val_embed_tensor1 = tf.placeholder(tf.float32, shape=[self.args.nbatch, self.args.m])
        self.val_embed_tensor2 = tf.placeholder(tf.float32, shape=[self.nval, self.args.m])

        self.p_dist = math_ops.add(
                    math_ops.reduce_sum(math_ops.square(self.val_embed_tensor1), axis=[1], keep_dims=True),
                    math_ops.reduce_sum(math_ops.square(array_ops.transpose(self.val_embed_tensor2)), axis=[0], keep_dims=True))-\
                2.0 * math_ops.matmul(self.val_embed_tensor1, array_ops.transpose(self.val_embed_tensor2)) # [batch_size, 1], [1, ndata],  [batch_size, ndata]

        self.p_dist = math_ops.maximum(self.p_dist, 0.0) # [batch_size, ndata] 
        self.p_dist = math_ops.multiply(self.p_dist, math_ops.to_float(math_ops.logical_not(math_ops.less_equal(self.p_dist, 0.0))))
        self.p_max_idx = tf.nn.top_k(-self.p_dist, k=2)[1] # [batch_size, 2] # get smallest 2

        self.logger.info("Model setting up train ends")

    def run_batch(self):
        '''
        Return : 
            following graph operations
        '''
        if self.args.ltype=='npair':
            batch_anc_img, batch_pos_img, batch_anc_label, batch_pos_label = self.dataset_dict['train'].next_batch(batch_size=self.args.nbatch)
            feed_dict = {self.anc_img : batch_anc_img, self.pos_img : batch_pos_img, self.label : batch_anc_label, self.istrain : True}
            batch_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)[1]
            return batch_loss

        else:# triplet
            batch_img, batch_label = self.dataset_dict['train'].next_batch(batch_size=self.args.nbatch)
            feed_dict = {self.img : batch_img, self.label : batch_label, self.istrain : True}
            batch_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)[1]
            return batch_loss

    def train(self, epoch, save_dir, board_dir):
        self.logger.info("Model training starts")

        self.writer = SummaryWriter(board_dir) 

        if self.args.ltype=='npair':
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.anc_img, output_gate=output_gate, batch_size=self.args.nbatch//2, dim=4, train_gate=self.istrain)
        else: # triplet
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.img, output_gate=output_gate, batch_size=self.args.nbatch, dim=4, train_gate=self.istrain)

        def eval_on_val():
            if self.args.ltype=='npair': self.val_embed = custom_apply_tf_op(inputs=self.val_image, output_gate=self.anc_embed)
            else: self.val_embed = custom_apply_tf_op(inputs=self.val_image, output_gate=self.embed_l2_norm) # triplet

            val_p1 = get_recall_at_1_efficient(
                            data=self.val_embed, label=self.val_label,\
                            input1_tensor=self.val_embed_tensor1, input2_tensor=self.val_embed_tensor2,\
                            idx_tensor=self.p_max_idx, session=self.sess)
            return val_p1

        val_p1 = eval_on_val()
        max_val_p1 = val_p1 
        self.save(0, save_dir)
        self.logger.info("Initial val p@1 = {}".format(val_p1))

        for epoch_ in tqdm_range(epoch):
            train_epoch_loss = 0
            for _ in tqdm_range(self.nbatch_train):
                batch_loss = self.run_batch_hash()
                train_epoch_loss += batch_loss	
            train_epoch_loss /= self.nbatch_train

            if train_epoch_loss!=train_epoch_loss: break # nan
            val_p1 = eval_on_val()
            self.logger.info("Epoch({}/{}) train loss = {} val p1 = {}".format(epoch_+1, epoch, train_epoch_loss, val_p1))	
            self.writer.add_summaries({"loss" : train_epoch_loss, "lr" : self.sess.run(self.lr), "p1" : val_p1}, epoch_+1)
            if max_val_p1 < val_p1:
                max_val_p1 = val_p1
                self.save(epoch_+1, save_dir)
        self.logger.info("Model training ends")

    def prepare_test(self):
        self.logger.info("Model preparing test")
        if self.args.ltype=='npair':
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.anc_img, output_gate=output_gate, batch_size=self.args.nbatch//2, dim=4, train_gate=self.istrain)
            self.val_embed = custom_apply_tf_op(inputs=self.val_image, output_gate=self.anc_embed)
            self.test_embed = custom_apply_tf_op(inputs=self.test_image, output_gate=self.anc_embed)
            self.train_embed = custom_apply_tf_op(inputs=self.train_image, output_gate=self.anc_embed)
        else: # triplet
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.img, output_gate=output_gate, batch_size=self.args.nbatch, dim=4, train_gate=self.istrain)
            self.val_embed = custom_apply_tf_op(inputs=self.val_image, output_gate=self.embed_l2_norm)
            self.test_embed = custom_apply_tf_op(inputs=self.test_image, output_gate=self.embed_l2_norm)
            self.train_embed = custom_apply_tf_op(inputs=self.train_image, output_gate=self.embed_l2_norm)

    def prepare_test2(self):
        self.te_te_distance = pairwise_distance_euclid_efficient(input1=self.test_embed, input2=self.test_embed, session=self.sess, batch_size=128)
        self.te_tr_distance = pairwise_distance_euclid_efficient(input1=self.test_embed, input2=self.train_embed, session=self.sess, batch_size=128)
        val_p_dist = pairwise_distance_euclid_efficient(input1=self.val_embed, input2=self.val_embed, session=self.sess, batch_size=self.args.nbatch)
        self.val_arg_sort = np.argsort(val_p_dist, axis=1)

    def set_info(self, te_te_distance, te_tr_distance, test_embed, train_embed):
        self.te_te_distance = te_te_distance
        self.te_tr_distance = te_tr_distance
        self.test_embed = test_embed
        self.train_embed = train_embed

    def test_metric(self, k_set):
        self.logger.info("Model testing metric starts")
        if not hasattr(self, 'te_tr_distance') and not hasattr(self, 'te_te_distance'):
            self.regen_session()
            self.te_te_distance = pairwise_distance_euclid_efficient(input1=self.test_embed, input2=self.test_embed, session=self.sess, batch_size=128)
            self.te_tr_distance = pairwise_distance_euclid_efficient(input1=self.test_embed, input2=self.train_embed, session=self.sess, batch_size=128)
        performance = evaluate_metric_te_tr(test_label=self.test_label, train_label=self.train_label, te_te_distance=self.te_te_distance, te_tr_distance=self.te_tr_distance, k_set=k_set, logger=self.logger) 
        self.regen_session()
        return performance

