import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from utils.general_class import ModelPlugin
from utils.evaluation import evaluate_hashtree_te_tr_sparsity
from utils.ortools_op import solve_maxmatching_soft_intraclass_multiselect, SolveMaxMatching
from utils.eval_op import get_nmi_suf_quick
from utils.np_op import activate_k_2D, plabel2subset, bws2label

from tfops.transform_op import apply_tf_op
from tfops.summary_op import SummaryWriter
from tfops.hash_dist import triplet_semihard_loss_hash, npairs_loss_hash,\
                            pairwise_distance_w_obj1, pairwise_similarity_w_obj1
from tfops.train_op import get_multi_train_op
from tfops.info_op import get_shape
from tfops.lr_op import DECAY_DICT, DECAY_PARAMS_DICT
from tfops.nets import conv1_32

import tensorflow as tf
slim = tf.contrib.slim

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
            self.label_list = [tf.placeholder(tf.int32, shape = [self.args.nbatch//2]) for idx in range(self.args.k)]
        else: # triplet
            self.img = tf.placeholder(tf.float32, shape = [self.args.nbatch, self.height, self.width, self.nchannel])
            self.istrain = tf.placeholder(tf.bool, shape= [])
            self.label_list = [tf.placeholder(tf.int32, shape = [self.args.nbatch]) for idx in range(self.args.k)]
        self.generate_sess()

        self.conv_net = conv1_32
        if self.args.ltype == 'npair':
            self.anc_last = tf.nn.relu(self.conv_net(self.anc_img, is_training=self.istrain, reuse=False)[0])
            self.pos_last = tf.nn.relu(self.conv_net(self.pos_img, is_training=self.istrain, reuse=True)[0])
        else:#triplet
            self.last = tf.nn.relu(self.conv_net(self.img, is_training=self.istrain, reuse=False)[0])
        self.logger.info("Model building ends")

    def set_info(self, val_arg_sort, te_te_distance, te_tr_distance):
        self.logger.info("Model setting info starts")
        self.val_arg_sort = val_arg_sort
        self.te_te_distance = te_te_distance
        self.te_tr_distance = te_tr_distance
        self.logger.info("Model setting info ends")
    
    def build_hash(self): 
        self.logger.info("Model building train hash starts")

        self.mcf = SolveMaxMatching(nworkers=self.args.nsclass, ntasks=self.args.d, k=1, pairwise_lamb=self.args.plamb2)

        with slim.arg_scope([slim.fully_connected], activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0005), biases_initializer=tf.zeros_initializer(), weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):
            if self.args.ltype=='triplet':
                # placeholder list
                self.objective_list = [tf.placeholder(dtype=tf.float32, shape=[self.args.nbatch, self.args.d], name="objective%d"%i) for i in range(self.args.k)] 
                self.embed_k_hash = self.last 
                with tf.variable_scope('Hash', reuse=False): self.embed_k_hash = slim.fully_connected(self.embed_k_hash, self.args.d*self.args.k, scope="fc1") # [batch, d*k]
                self.embed_k_hash_list = tf.split(self.embed_k_hash, num_or_size_splits=self.args.k, axis=1) # list(k*[batch, d])
                self.embed_k_hash_l2_norm_list = [tf.nn.l2_normalize(v, dim=-1) for v in self.embed_k_hash_list]# list(k*[batch,d]), each l2 normalize
                self.pairwise_distance = pairwise_distance_w_obj1

                self.loss_hash = 0
                for idx in range(self.args.k): self.loss_hash += triplet_semihard_loss_hash(labels=self.label_list[idx], embeddings=self.embed_k_hash_l2_norm_list[idx], objectives=self.objective_list[idx], pairwise_distance=self.pairwise_distance, margin=self.args.param)
            else: 
                self.objective_list = [tf.placeholder(dtype=tf.float32, shape=[self.args.nbatch//2, self.args.d], name="objective%d"%i) for i in range(self.args.k)] 
                self.anc_embed_k_hash = self.anc_last
                self.pos_embed_k_hash = self.pos_last
                with tf.variable_scope('Hash', reuse=False): self.anc_embed_k_hash = slim.fully_connected(self.anc_embed_k_hash, self.args.d*self.args.k, scope="fc1")
                with tf.variable_scope('Hash', reuse=True): self.pos_embed_k_hash = slim.fully_connected(self.pos_embed_k_hash, self.args.d*self.args.k, scope="fc1")
                self.anc_embed_k_hash_list = tf.split(self.anc_embed_k_hash, num_or_size_splits=self.args.k, axis=1) # list(k*[batch, d])
                self.pos_embed_k_hash_list = tf.split(self.pos_embed_k_hash, num_or_size_splits=self.args.k, axis=1) # list(k*[batch, d])
                self.similarity_func = pairwise_similarity_w_obj1

                self.loss_hash = 0
                for idx in range(self.args.k):
                    self.loss_hash += npairs_loss_hash(labels=self.label_list[idx], embeddings_anchor=self.anc_embed_k_hash_list[idx], embeddings_positive=self.pos_embed_k_hash_list[idx],\
                                            objective=self.objective_list[idx], similarity_func=self.similarity_func, reg_lambda=self.args.param)

        self.EMBED_K_HASH_LIST = self.anc_embed_k_hash_list if self.args.ltype=='npair' else self.embed_k_hash_l2_norm_list
        self.tree_idx_set = [tf.nn.top_k(v, k=self.args.d)[1] for v in self.EMBED_K_HASH_LIST]# k*[batch_size, d]
        self.tree_idx = tf.transpose(tf.stack(self.tree_idx_set, axis=0), [1, 0, 2]) # [batch_size, k, d]

        self.logger.info("Model building train hash ends")

    def set_up_train_hash(self):
        self.logger.info("Model setting up train hash starts")

        decay_func = DECAY_DICT[self.args.dtype]
        self.lr, update_step_op = decay_func(**DECAY_PARAMS_DICT[self.args.dtype][self.args.nbatch][self.args.dptype])

        update_ops = tf.get_collection("update_ops")
        var_slow_list, var_fast_list = list(), list()
        for var in tf.trainable_variables():
            if 'Hash' in var.name: var_fast_list.append(var)
            else: var_slow_list.append(var)

        with tf.control_dependencies(update_ops+[update_step_op]): self.train_op_hash = get_multi_train_op(tf.train.AdamOptimizer, self.loss_hash, [0.1*self.lr, self.lr], [var_slow_list, var_fast_list])

        self.idx_convert = list()
        tmp = 1
        for i in range(self.args.k):
            self.idx_convert.append(tmp)
            tmp*=self.args.d
        self.idx_convert = np.array(self.idx_convert)[::-1] # [d**(k-1),...,1]
        self.idx_convert = tf.constant(self.idx_convert, dtype=tf.int32) # tensor [k]

        self.max_k_idx = tf.add(
                tf.reduce_sum(tf.multiply(self.tree_idx[:,:-1,0], self.idx_convert[:-1]), axis=-1, keep_dims=True),\
                self.tree_idx[:, -1, :self.args.sk]) # [batch_size, sk]

        if self.args.ltype=='npair': assert get_shape(self.max_k_idx) == [self.args.nbatch//2, self.args.sk], "Wrong max_k_idx shape"
        else: assert get_shape(self.max_k_idx) == [self.args.nbatch, self.args.sk], "Wrong max_k_idx shape"

        self.logger.info("Model setting up train hash ends")

    def run_batch_hash(self):
        if self.args.ltype=='npair':
            batch_anc_img, batch_pos_img, batch_anc_label, batch_pos_label = self.dataset_dict['train'].next_batch(batch_size=self.args.nbatch)
            feed_dict = {self.anc_img : batch_anc_img, self.pos_img : batch_pos_img, self.istrain : True}

            objective_list, plabel_list = list(), list()

            anc_unary_list, pos_unary_list = self.sess.run([self.anc_embed_k_hash_list, self.pos_embed_k_hash_list], feed_dict=feed_dict) # k*[nbatch//2, d]
            unary_list = [0.5*(anc_unary_list[k_idx]+pos_unary_list[k_idx]) for k_idx in range(self.args.k)] # k*[nbatch//2, d]
            unary_list = [np.mean(np.reshape(v, [self.args.nsclass, -1, self.args.d]), axis=1) for v in unary_list] # k*[nsclass, d]

            for k_idx in range(self.args.k):
                unary = unary_list[k_idx] # [nsclass, d]
                plabel = np.zeros(self.args.nsclass, dtype=np.int32) # [nsclass]
                if k_idx!=0:
                    prev_plabel = plabel_list[-1]
                    prev_objective = objective_list[-1]
                    for i in range(self.args.nsclass):
                        plabel[i] = prev_plabel[i]*self.args.d+np.argmax(prev_objective[i])
                
                objective = np.zeros([self.args.nsclass, self.args.d], dtype=np.float32) # [nsclass, d]

                if k_idx==0: results = self.mcf.solve(unary)
                elif k_idx==self.args.k-1: results = solve_maxmatching_soft_intraclass_multiselect(array=unary, k=self.args.sk, labels=plabel, plamb1=self.args.plamb1, plamb2=self.args.plamb2)
                else: results = solve_maxmatching_soft_intraclass_multiselect(array=unary, k=1, labels=plabel, plamb1=self.args.plamb1, plamb2=self.args.plamb2)

                for i,j in results: objective[i][j]=1

                plabel_list.append(plabel)
                objective_list.append(objective)

            objective_list  = [np.reshape(np.transpose(np.tile(np.transpose(v, [1,0]), [self.args.nbatch//(2*self.args.nsclass), 1]), [1,0]), [self.args.nbatch//2, self.args.d]) for v in objective_list] # k*[nsclass, d] => k*[batch_size//2, d]
            plabel_list = [np.reshape(np.tile(np.expand_dims(v, axis=-1), [1, self.args.nbatch//(2*self.args.nsclass)]), [-1]) for v in plabel_list ] # k*[nsclass] => k*[batch_size//2]
            for k_idx in range(self.args.k):
                feed_dict[self.objective_list[k_idx]] = objective_list[k_idx]
                if k_idx==self.args.k-1: feed_dict[self.label_list[k_idx]] = bws2label(objective=objective_list[k_idx], sparsity=self.args.sk) if self.args.label == 'dynamic' else batch_anc_label # [nbatch//2]
                else: feed_dict[self.label_list[k_idx]] = np.argmax(objective_list[k_idx], axis=1) if self.args.label == 'dynamic' else batch_anc_label # [nbatch//2]

            batch_loss_hash = self.sess.run([self.train_op_hash, self.loss_hash], feed_dict=feed_dict)[1]
            return batch_loss_hash
        else:
            batch_img, batch_label = self.dataset_dict['train'].next_batch(batch_size=self.args.nbatch)

            feed_dict = {self.img : batch_img, self.istrain : True}
            objective_list, plabel_list = list(), list()

            unary_list = self.sess.run(self.embed_k_hash_l2_norm_list, feed_dict=feed_dict) # k*[nbatch, d]
            unary_list = [np.mean(np.reshape(v, [self.args.nsclass, -1, self.args.d]), axis=1) for v in unary_list] # k*[nsclass, d]
            for k_idx in range(self.args.k):
                unary = unary_list[k_idx] # [nsclass, d]
                plabel = np.zeros(self.args.nsclass, dtype=np.int32) # [nsclass]
                if k_idx!=0:
                    prev_plabel = plabel_list[-1]
                    prev_objective = objective_list[-1]
                    for i in range(self.args.nsclass):
                        plabel[i] = prev_plabel[i]*self.args.d+np.argmax(prev_objective[i])

                objective = np.zeros([self.args.nsclass, self.args.d], dtype=np.float32) # [nsclass, d]

                if k_idx==0: results = self.mcf.solve(unary)
                elif k_idx==self.args.k-1: results = solve_maxmatching_soft_intraclass_multiselect(array=unary, k=self.args.sk, labels=plabel, plamb1=self.args.plamb1, plamb2=self.args.plamb2)
                else: results = solve_maxmatching_soft_intraclass_multiselect(array=unary, k=1, labels=plabel, plamb1=self.args.plamb1, plamb2=self.args.plamb2)

                for i,j in results: objective[i][j]=1

                plabel_list.append(plabel)
                objective_list.append(objective)

            objective_list = [np.reshape(np.transpose(np.tile(np.transpose(v, [1,0]), [self.args.nbatch//self.args.nsclass , 1]), [1,0]), [self.args.nbatch, -1]) for v in objective_list ] # k*[nsclass, d] => k*[batch_size, d]
            plabel_list = [np.reshape(np.tile(np.expand_dims(v, axis=-1), [1, self.args.nbatch//self.args.nsclass]), [-1]) for v in plabel_list ] # k*[nsclass] => k*[batch_size]

            for k_idx in range(self.args.k):
                feed_dict[self.objective_list[k_idx]] = objective_list[k_idx]

                if k_idx==self.args.k-1: feed_dict[self.label_list[k_idx]] = bws2label(objective=objective_list[k_idx], sparsity=self.args.sk) if self.args.label == 'dynamic' else batch_label # [nbatch]
                else: feed_dict[self.label_list[k_idx]] = np.argmax(objective_list[k_idx], axis=1) if self.args.label == 'dynamic' else batch_label # [nbatch]

            batch_loss_hash = self.sess.run([self.train_op_hash, self.loss_hash], feed_dict=feed_dict)[1]
            return batch_loss_hash

    def train_hash(self, epoch, save_dir, board_dir):
        self.logger.info("Model training starts")
        self.writer = SummaryWriter(board_dir) 
        self.logger.info("initial_lr : {}".format(self.sess.run(self.lr)))
        if self.args.ltype=='npair':
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.anc_img, output_gate=output_gate, batch_size=self.args.nbatch//2, dim=4, train_gate=self.istrain)
        else: # triplet
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.img, output_gate=output_gate, batch_size=self.args.nbatch, dim=4, train_gate=self.istrain)

        def eval_on_val():
            val_max_k_idx = custom_apply_tf_op(inputs=self.val_image, output_gate=self.max_k_idx) # [nval, sk]
            val_nmi, val_suf = get_nmi_suf_quick(index_array=val_max_k_idx, label_array=self.val_label, ncluster=self.args.d**self.args.k, nlabel=self.ncls_val)
            nsuccess=0
            for i in range(self.nval):
                for j in self.val_arg_sort[i]:
                    if i==j: continue
                    if len(set(val_max_k_idx[j])&set(val_max_k_idx[i]))>0:
                        if self.val_label[i]==self.val_label[j]: nsuccess+=1
                        break
            val_p1 = nsuccess/self.nval
            return val_suf, val_nmi, val_p1 

        val_suf, val_nmi, val_p1 = eval_on_val()
        max_val_p1=val_p1
        self.logger.info("Initial val_suf = {} val_nmi = {} val_p1 = {}".format(val_suf, val_nmi, val_p1))	
        self.save(0, save_dir)
        for epoch_ in range(epoch):
            train_epoch_loss = 0
            for _ in range(self.nbatch_train):
                batch_loss = self.run_batch_hash()
                train_epoch_loss += batch_loss	
            train_epoch_loss /= self.nbatch_train
            val_suf, val_nmi, val_p1 = eval_on_val()

            self.logger.info("Epoch({}/{}) train loss = {} val suf = {} val nmi = {} val p1 = {}".format(epoch_+1, epoch, train_epoch_loss, val_suf, val_nmi, val_p1))	
            self.writer.add_summaries({"loss" : train_epoch_loss, "lr" : self.sess.run(self.lr), "suf" : val_suf, "nmi" : val_nmi, "p1" : val_p1}, epoch_+1)
            if max_val_p1 < val_p1:
                max_val_p1 = val_p1
                self.save(epoch_+1, save_dir)
        self.logger.info("Model training ends")

    def prepare_test_hash(self):
        self.logger.info("Model preparing test")
        if self.args.ltype=='npair':
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.anc_img, output_gate=output_gate, batch_size=self.args.nbatch//2, dim=4, train_gate=self.istrain)
        else: # triplet
            def custom_apply_tf_op(inputs, output_gate): return apply_tf_op(inputs=inputs, session=self.sess, input_gate=self.img, output_gate=output_gate, batch_size=self.args.nbatch, dim=4, train_gate=self.istrain)
        self.test_max_k_idx = custom_apply_tf_op(inputs=self.test_image, output_gate=self.max_k_idx) # [ntest, sk]
        self.test_tree_idx = custom_apply_tf_op(inputs=self.test_image, output_gate=self.tree_idx) # [ntest, k, d]

        self.train_max_k_idx = custom_apply_tf_op(inputs=self.train_image, output_gate=self.max_k_idx) # [ntrain, sk]
        self.train_tree_idx = custom_apply_tf_op(inputs=self.train_image, output_gate=self.tree_idx) # [ntrain, k, d]


    def test_hash_metric(self, k_set):
        self.logger.info("Model testing k hash starts")

        performance = evaluate_hashtree_te_tr_sparsity(te_tr_distance=self.te_tr_distance, te_te_distance=self.te_te_distance,\
                                                       train_tree_idx=self.train_tree_idx, test_tree_idx=self.test_tree_idx,\
                                                       train_max_k_idx=self.train_max_k_idx, test_max_k_idx=self.test_max_k_idx,\
                                                       train_label=self.train_label, test_label=self.test_label,\
                                                       ncls_train=self.ncls_train, ncls_test=self.ncls_test, k_set=k_set, logger=self.logger)

        self.logger.info("Model testing k hash ends")
        return performance

