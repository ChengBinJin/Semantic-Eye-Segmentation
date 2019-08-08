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
        self.is_train = is_train
        self.log_dir = log_dir
        self.resize_factor = resize_factor
        self.name = name
        self.layers = [2, 2, 2, 2]
        self._ops = list()
        self.tb_lr = None

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, logDir=self.log_dir, isTrain=self.is_train, name=self.name)

        self._build_graph()
        # self.init_tensorboard()
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None)

    def _build_graph(self):
        self.train_mode = tf.compat.v1.placeholder(dtype=tf.bool, name='train_mode_ph')

        # Initialize TFRecoder reader
        train_reader = ReaderIdentity(tfrecords_file=self.data_path[0],
                                      decode_img_shape=self.decode_img_shape,
                                      batch_size=self.batch_size,
                                      name='train')

        # Random batch for training
        self.img_train, self.cls_train = train_reader.shuffle_batch()

        # Network forward for training
        self.pred_train = self.forward_network(input_img=self.normalize(self.img_train), reuse=False)
        self.pred_train_cls = tf.math.argmax(self.pred_train, axis=-1)

        # Data loss
        self.data_loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.pred_train,
            labels=self.convert_one_hot(self.cls_train)))

        # Regularization term
        variables = self.get_regularization_variables()
        self.reg_term = self.weight_decay * tf.math.reduce_mean([tf.nn.l2_loss(variable) for variable in variables])

        # Total loss
        self.total_loss = self.data_loss + self.reg_term

        # Optimizer
        self.train_op = self.init_optimizer(loss=self.total_loss)

    def init_optimizer(self, loss, name='Adam'):
        with tf.compat.v1.variable_scope(name):
            global_step = tf.Variable(0., dtype=tf.float32, trainable=False)
            start_learning_rate = self.lr
            end_leanring_rate = self.lr * 0.001
            start_decay_step = int(self.total_steps * 0.5)
            decay_steps = self.total_steps - start_decay_step

            learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                      tf.compat.v1.train.polynomial_decay(learning_rate=start_learning_rate,
                                                                          global_step=(global_step - start_decay_step),
                                                                          decay_steps=decay_steps,
                                                                          end_learning_rate=end_leanring_rate,
                                                                          power=1.0), start_learning_rate))
            self.tb_lr = tf.compat.v1.summary.scalar('leanring_rate', learning_rate)

            learn_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.99).minimize(
                loss, global_step=global_step)

        return learn_step


    def forward_network(self, input_img, padding='SAME', reuse=False):
        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            tf_utils.print_activations(input_img, logger=self.logger)
            inputs = self.conv2d_fixed_padding(inputs=input_img, filters=64, kernel_size=7, strides=2, name='conv1')
            inputs = tf_utils.max_pool(inputs, name='3x3_maxpool', ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                       logger=self.logger)

            inputs = self.block_layer(inputs=inputs, filters=64, block_fn=self.bottleneck_block, blocks=self.layers[0],
                                      strides=1, train_mode=self.train_mode, name='block_layer1')
            inputs = self.block_layer(inputs=inputs, filters=128, block_fn=self.bottleneck_block, blocks=self.layers[1],
                                      strides=2, train_mode=self.train_mode, name='block_layer2')
            inputs = self.block_layer(inputs=inputs, filters=256, block_fn=self.bottleneck_block, blocks=self.layers[2],
                                      strides=2, train_mode=self.train_mode, name='block_layer3')
            inputs = self.block_layer(inputs=inputs, filters=512, block_fn=self.bottleneck_block, blocks=self.layers[3],
                                      strides=2, train_mode=self.train_mode, name='block_layer4')

            inputs = tf_utils.norm(inputs, name='before_gap_batch_norm', _type='batch', _ops=self._ops,
                                   is_train=self.train_mode, logger=self.logger)
            inputs = tf_utils.relu(inputs, name='before_gap_relu', logger=self.logger)
            _, h, w, _ = inputs.get_shape().as_list()
            inputs = tf_utils.avg_pool(inputs, name='gap', ksize=[1, h, w, 1], strides=[1, 1, 1, 1], logger=self.logger)

            inputs = tf_utils.flatten(inputs, name='flatten', logger=self.logger)
            logits = tf_utils.linear(inputs, self.num_classes, name='logits')

            return logits

    def block_layer(self, inputs, filters, block_fn, blocks, strides, train_mode, name):
        # Only the first block per block_layer uses projection_shortcut and strides
        inputs = block_fn(inputs, filters, train_mode, self.projection_shortcut, strides, name + '_1')

        for num_iter in range(1, blocks):
            inputs = block_fn(inputs, filters, train_mode, None, 1, name=(name + '_' + str(num_iter + 1)))

        return tf.identity(inputs, name)

    def bottleneck_block(self, inputs, filters, train_mode, projection_shortcut, strides, name):
        with tf.compat.v1.variable_scope(name):
            shortcut = inputs

            # norm(x, name, _type, _ops, is_train=True, is_print=True, logger=None)
            inputs = tf_utils.norm(inputs, name='batch_norm_0', _type='batch', _ops=self._ops,
                                   is_train=train_mode, logger=self.logger)
            inputs = tf_utils.relu(inputs, name='relu_0', logger=self.logger)

            # The projection shortcut shouldcome after the first batch norm and ReLU since it perofrms a 1x1 convolution.
            if projection_shortcut is not None:
                shortcut = self.projection_shortcut(inputs=inputs, filters_out=filters, strides=strides, name='conv_projection')

            inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, name='conv_0')

            inputs = tf_utils.norm(inputs, name='batch_norm_1', _type='batch', _ops=self._ops,
                                   is_train=train_mode, logger=self.logger)
            inputs = tf_utils.relu(inputs, name='relu_1', logger=self.logger)
            inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, name='conv_1')

            output = tf.identity(inputs + shortcut, name=(name + '_output'))
            tf_utils.print_activations(output, logger=self.logger)

            return output

    def projection_shortcut(self, inputs, filters_out, strides, name):
        inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides, name=name)
        return inputs

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, name):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size)

        inputs = tf_utils.conv2d(inputs, output_dim=filters, k_h=kernel_size, k_w=kernel_size,
                                 d_h=strides, d_w=strides, initializer='He', name=name,
                                 padding=('SAME' if strides == 1 else 'VALID'), logger=self.logger)
        return inputs

    @staticmethod
    def get_regularization_variables():
        # We exclude 'bias', 'beta' and 'gamma' in batch normalization
        variables = [variable for variable in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
                     if ('bias' not in variable.name) and
                     ('beta' not in variable.name) and
                     ('gamma' not in variable.name)]

        return variables

    def convert_one_hot(self, data):
        shape = data.get_shape().as_list()
        data = tf.dtypes.cast(data, dtype=tf.uint8)
        data = tf.one_hot(data, depth=self.num_classes, name='one_hot')
        data = tf.reshape(data, shape=[*shape[:3], self.num_classes])
        return data

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
