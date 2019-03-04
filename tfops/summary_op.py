import tensorflow as tf
import os 

class SummaryWriter:
    def __init__(self, save_path):
        if not os.path.exists(save_path): os.makedirs(save_path) 
        self.writer = tf.summary.FileWriter(save_path)

    def add_summary(self, tag, simple_value, global_step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=simple_value)])
        self.writer.add_summary(summary, global_step)

    def add_summaries(self, dict_, global_step):
        for key in dict_.keys(): self.add_summary(tag=str(key), simple_value=dict_[key], global_step=global_step)

