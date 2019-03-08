import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.logger_op import LoggerManager
from tfops.init_op import rest_initializer

import tensorflow as tf
import glob

class ModelPlugin:
    def __init__(self, train_dataset, val_dataset, test_dataset, logfilepath, args):
        self.args = args

        self.logfilepath = logfilepath
        self.logger = LoggerManager(self.logfilepath, __name__)

        self.dataset_dict = dict()
        self.set_train_dataset(train_dataset) 
        self.set_val_dataset(val_dataset) 
        self.set_test_dataset(test_dataset) 

    def set_train_dataset(self, train_dataset):
        self.logger.info("Setting train_dataset starts")
        self.train_dataset = train_dataset
        self.dataset_dict['train'] = self.train_dataset
        self.train_image = self.dataset_dict['train'].image
        self.train_label = self.dataset_dict['train'].label
        self.ntrain, self.height, self.width, self.nchannel = self.train_image.shape
        self.ncls_train = self.train_dataset.nclass
        self.nbatch_train = self.ntrain//self.args.nbatch
        self.logger.info("Setting train_dataset ends")

    def set_test_dataset(self, test_dataset):
        self.logger.info("Setting test_dataset starts")
        self.test_dataset = test_dataset
        self.dataset_dict['test'] = self.test_dataset
        self.test_image = self.dataset_dict['test'].image 
        self.test_label = self.dataset_dict['test'].label
        self.ntest = self.test_dataset.ndata
        self.ncls_test = self.test_dataset.nclass
        self.nbatch_test = self.ntest//self.args.nbatch
        self.logger.info("Setting test_dataset ends")

    def set_val_dataset(self, val_dataset):
        self.logger.info("Setting val_dataset starts")
        self.val_dataset = val_dataset
        self.dataset_dict['val'] = self.val_dataset
        self.val_image = self.dataset_dict['val'].image 
        self.val_label = self.dataset_dict['val'].label
        self.nval = self.val_dataset.ndata
        self.ncls_val = self.val_dataset.nclass
        self.nbatch_val = self.nval//self.args.nbatch
        self.logger.info("Setting val_dataset ends")

    def build(self, *args, **kwargs):
        """Builds the neural networks"""
        raise NotImplementedError('`build` is not implemented for model class {}'.format(self.__class__.__name__))

    def set_up_train(self, *args, **kwargs):
        """Builds the neural networks"""
        raise NotImplementedError('`set_up_train` is not implemented for model class {}'.format(self.__class__.__name__))

    def train(self, *args, **kwargs):
        """train the neural networks"""
        raise NotImplementedError('`train` is not implemented for model class {}'.format(self.__class__.__name__))

    def generate_sess(self):
        try: self.sess
        except AttributeError:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess=tf.Session(config=config)

    def initialize(self):
        '''Initialize uninitialized variables'''
        self.logger.info("Model initialization starts")
        rest_initializer(self.sess) 
        self.start_epoch = 0
        self.logger.info("Model initialization ends")

    def save(self, global_step, save_dir, reset_option=True):
        self.logger.info("Model save starts")
        if reset_option:
            for f in glob.glob(save_dir+'*'): os.remove(f)
        saver=tf.train.Saver(max_to_keep = 5)
        saver.save(self.sess, os.path.join(save_dir, 'model'), global_step = global_step)
        self.logger.info("Model save in %s"%save_dir)
        self.logger.info("Model save ends")

    def restore(self, save_dir, restore_iter=-1):
        """Restore all variables in graph with the latest version"""
        self.logger.info("Restoring model starts...")
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(save_dir)

        if restore_iter==-1:
            self.start_iter = int(os.path.basename(checkpoint)[len('model')+1:])
        else:
            self.start_iter = restore_iter
            checkpoint = save_dir+'model-%d'%restore_iter
        self.logger.info("Restoring from {}".format(checkpoint))
        self.generate_sess()
        saver.restore(self.sess, checkpoint)
        self.logger.info("Restoring model done.")        

    def regen_session(self):
        self.generate_sess()
        tf.reset_default_graph()
        self.sess.close()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess=tf.Session(config=config)

    def delete(self):
        tf.reset_default_graph()
        self.logger.remove()
        del self.logger
