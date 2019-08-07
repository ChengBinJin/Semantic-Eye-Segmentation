# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Eye Generation Challenge
# Iris Identification
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import logging
import tensorflow as tf

import utils as utils
import tensorflow_utils as tf_utils
from reader import ReaderIdentity

class ResNet18(object):
    def __init__(self, decode_img_shape=(320, 200, 1), num_classes=152, data_path=(None, None), batch_size=1, lr=1e-3,
                 weight_decay=1e-4, total_iters=2e5, is_train=True, log_dir=None, resize_factor=0.5, name='ResNet18'):
        self.decode_img_shape = decode_img_shape
        self.num_classes = num_classes
        self.data_path = data_path
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_steps = total_iters
        self.start_decay_step = int(self.total_steps * 0.5)
        self.decay_steps = self.total_steps - self.start_decay_step
        self.is_train = is_train
        self.log_dir = log_dir
        self.resize_factor = resize_factor
        self.name = name
        self.layer = [2, 2, 2, 2]

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, logDir=self.log_dir, isTrain=self.is_train, name=self.name)

        # self._build_graph()
        # self.init_tensorboard()
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None)

    def _build_graph(self):
        # Initialize TFRecoder reader
        train_reader = ReaderIdentity(tfrecords_file=self.data_path[0],
                                      decode_img_shape=self.decode_img_shape,
                                      batch_size=self.batch_size,
                                      name='train')

        # Random batch for training
        self.img_train, self.cls_train = train_reader.shuffle_batch()

        # Network forward for training
        self.pred_train = self.forward_network(input_img=self.normalize(self.img_train), reuse=False)

    def forward_network(self, input_img, padding='SAME', reuse=False):
        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            tf_utils.print_activations(input_img, logger=self.logger)
            inputs = self.conv2d_fixed_padding(inputs=input_img, filters=64, kernel_size=7, strides=2, name='conv1')
            inputs = tf_utils.max_pool(inputs, name='3x3_maxpool', ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                       logger=self.logger)



    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, name):
        with tf.compat.v1.variable_scope(name):
            if strides > 1:
                inputs = self.fixed_padding(inputs, kernel_size)

            inputs = tf_utils.conv2d(inputs, output_dim=filters, k_h=kernel_size, k_w=kernel_size,
                                     d_h=strides, d_w=strides, initializer='He',
                                     padding=('SAME' if strides == 1 else 'VALID'), logger=self.logger)
            return inputs

    @staticmethod
    def fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_start = pad_total // 2
        pad_end = pad_total - pad_start
        inputs = tf.pad(inputs, [[0, 0], [pad_start, pad_end], [pad_start, pad_end], [0, 0]])
        return inputs

    @staticmethod
    def normalize(data):
        return data / 127.5 - 1.0
