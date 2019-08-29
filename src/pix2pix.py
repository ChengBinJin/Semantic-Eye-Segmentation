# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Synthetic Eye Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import logging
import tensorflow as tf

import utils as utils
import tensorflow_utils as tf_utils
from reader import Reader


class Pix2pix(object):
    def __init__(self, decode_img_shape=(320, 400, 1), output_shape=(320, 200, 1), num_classes=4, data_path=(None, None),
                 batch_size=1, lr=2e-4, total_iters=2e5, is_train=True, log_dir=None, resize_factor=0.5, lambda_1=10.,
                 name='pix2pix'):
        self.decode_img_shape = decode_img_shape
        self.input_shape = output_shape
        self.output_shape = output_shape
        self.num_classes = num_classes
        self.is_train = is_train
        self.resize_factor = resize_factor
        self.labmda_1 = lambda_1
        self.gen_c = [64, 128, 256, 512, 512, 512, 512, 512,
                      512, 512, 512, 512, 256, 128, 64, self.output_shape[2]]
        self.dis_c = [64, 128, 256, 512, 1]
        # self.conv_dims = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64, 1]

        self.data_path = data_path
        self.batch_size = batch_size
        self.lr = lr
        self.total_steps = total_iters
        self.start_decay_step = int(self.total_steps * 0.5)
        self.decay_steps = self.total_steps - self.start_decay_step
        self.log_dir = log_dir
        self.name = name
        self.tb_lr = None

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, logDir=self.log_dir, isTrain=self.is_train, name=self.name)

        self._build_graph()         # main graph
        self._init_tensorboard()    # tensorboard
        tf_utils.show_all_variables(logger=self.logger if self.is_train else None)

    def _build_graph(self):
        self.rate_ph = tf.compat.v1.placeholder(tf.float32, name='keep_prob_ph')

        # Initialize TFRecord reader
        train_reader = Reader(tfrecordsFile=self.data_path[0],
                              decodeImgShape=self.decode_img_shape,
                              imgShape=self.input_shape,
                              batchSize=self.batch_size,
                              name='train')

        # Initialize generator & discriminator
        self.G_gen = Generator(name='G', gen_c=self.gen_c, norm='instance', logger=self.logger, _ops=None)
        self.D_dis = Discriminator(name='D', dis_c=self.dis_c, norm='instance', logger=self.logger,  _ops=None)

        # Random batch for training
        self.img_train, self.seg_img_train = train_reader.shuffle_batch()

        self.g_sample = self.G_gen(self.transform_seg(self.seg_img_train), self.rate_ph)
        self.real_pair = tf.concat([self.transform_seg(self.seg_img_train), self.transform_img(self.img_train)], axis=3)
        self.fake_pair = tf.concat([self.transform_seg(self.seg_img_train), self.g_sample], axis=3)

    def _init_tensorboard(self):
        print("Hello _init_tensorboard!")

    def transform_seg(self, img):
        # label 0~3
        img = img * 255. / (self.num_classes - 1)
        img = img / 127.5 - 1.
        return img

    @staticmethod
    def transform_img(img):
        return img / 127.5 - 1.


class Generator(object):
    def __init__(self, name=None, gen_c=None, norm='instance', logger=None, _ops=None):
        self.name = name
        self.gen_c = gen_c
        self.norm = norm
        self.logger = logger
        self._ops = _ops
        self.reuse = False

    def __call__(self, x, keep_rate=0.5):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x, logger=self.logger)

            # E0: (320, 200) -> (160, 100)
            e0_conv2d = tf_utils.conv2d(x, self.gen_c[0], name='e0_conv2d')
            e0_lrelu = tf_utils.lrelu(e0_conv2d, name='e0_lrelu')

            # E1: (160, 100) -> (80, 50)
            e1_conv2d = tf_utils.conv2d(e0_lrelu, self.gen_c[1], name='e1_conv2d')
            e1_batchnorm = tf_utils.norm(e1_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e1_norm')
            e1_lrelu = tf_utils.lrelu(e1_batchnorm, name='e1_lrelu')

            # E2: (80, 50) -> (40, 25)
            e2_conv2d = tf_utils.conv2d(e1_lrelu, self.gen_c[2], name='e2_conv2d')
            e2_batchnorm = tf_utils.norm(e2_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e2_norm')
            e2_lrelu = tf_utils.lrelu(e2_batchnorm, name='e2_lrelu')

            # E3: (40, 25) -> (20, 13)
            e3_conv2d = tf_utils.conv2d(e2_lrelu, self.gen_c[3], name='e3_conv2d')
            e3_batchnorm = tf_utils.norm(e3_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e3_norm')
            e3_lrelu = tf_utils.lrelu(e3_batchnorm, name='e3_lrelu')

            # E4: (20, 13) -> (10, 7)
            e4_conv2d = tf_utils.conv2d(e3_lrelu, self.gen_c[4], name='e4_conv2d')
            e4_batchnorm = tf_utils.norm(e4_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e4_norm')
            e4_lrelu = tf_utils.lrelu(e4_batchnorm, name='e4_lrelu')

            # E5: (10, 7) -> (5, 4)
            e5_conv2d = tf_utils.conv2d(e4_lrelu, self.gen_c[5], name='e5_conv2d')
            e5_batchnorm = tf_utils.norm(e5_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e5_norm')
            e5_lrelu = tf_utils.lrelu(e5_batchnorm, name='e5_lrelu')

            # E6: (5, 4) -> (3, 2)
            e6_conv2d = tf_utils.conv2d(e5_lrelu, self.gen_c[6], name='e6_conv2d')
            e6_batchnorm = tf_utils.norm(e6_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e6_norm')
            e6_lrelu = tf_utils.lrelu(e6_batchnorm, name='e6_lrelu')

            # E7: (3, 2) -> (2, 1)
            e7_conv2d = tf_utils.conv2d(e6_lrelu, self.gen_c[7], name='e7_conv2d')
            e7_batchnorm = tf_utils.norm(e7_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='e7_norm')
            e7_relu = tf_utils.relu(e7_batchnorm, name='e7_relu')

            # D0: (2, 1) -> (3, 2)
            # Stage1: (2, 1) -> (4, 2)
            d0_deconv = tf_utils.deconv2d(e7_relu, self.gen_c[8], name='d0_deconv2d')
            # Stage2: (4, 2) -> (3, 2)
            shapeA = e6_conv2d.get_shape().as_list()[1]
            shapeB = d0_deconv.get_shape().as_list()[1] - e6_conv2d.get_shape().as_list()[1]
            d0_split, _ = tf.split(d0_deconv, [shapeA, shapeB], axis=1, name='d0_split')
            tf_utils.print_activations(d0_split, logger=self.logger)
            # Stage3: Batch norm, concatenation, and relu
            d0_batchnorm = tf_utils.norm(d0_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d0_norm')
            d0_drop = tf.nn.dropout(d0_batchnorm, keep_prob=keep_rate, name='d0_dropout')
            d0_concat = tf.concat([d0_drop, e6_batchnorm], axis=3, name='d0_concat')
            d0_relu = tf_utils.relu(d0_concat, name='d0_relu')

            # D1: (3, 2) -> (5, 4)
            # Stage1: (3, 2) -> (6, 4)
            d1_deconv = tf_utils.deconv2d(d0_relu, self.gen_c[9], name='d1_deconv2d')
            # Stage2: (6, 4) -> (5, 4)
            shapeA = e5_batchnorm.get_shape().as_list()[1]
            shapeB = d1_deconv.get_shape().as_list()[1] - e5_batchnorm.get_shape().as_list()[1]
            d1_split, _ = tf.split(d1_deconv, [shapeA, shapeB], axis=1, name='d1_split')
            tf_utils.print_activations(d1_split, logger=self.logger)
            # Stage3: Batch norm, concatenation, and relu
            d1_batchnorm = tf_utils.norm(d1_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d1_norm')
            d1_drop = tf.nn.dropout(d1_batchnorm, keep_prob=keep_rate, name='d1_dropout')
            d1_concat = tf.concat([d1_drop, e5_batchnorm], axis=3, name='d1_concat')
            d1_relu = tf_utils.relu(d1_concat, name='d1_relu')

            # D2: (5, 4) -> (10, 7)
            # Stage1: (5, 4) -> (10, 8)
            d2_deconv = tf_utils.deconv2d(d1_relu, self.gen_c[10], name='d2_deconv2d')
            # Stage2: (10, 8) -> (10, 7)
            shapeA = e4_batchnorm.get_shape().as_list()[2]
            shapeB = d2_deconv.get_shape().as_list()[2] - e4_batchnorm.get_shape().as_list()[2]
            d2_split, _ = tf.split(d2_deconv, [shapeA, shapeB], axis=2, name='d2_split')
            tf_utils.print_activations(d2_split, logger=self.logger)
            # Stage3: Batch norm, concatenation, and relu
            d2_batchnorm = tf_utils.norm(d2_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d2_norm')
            d2_drop = tf.nn.dropout(d2_batchnorm, keep_prob=keep_rate, name='d2_dropout')
            d2_concat = tf.concat([d2_drop, e4_batchnorm], axis=3, name='d2_concat')
            d2_relu = tf_utils.relu(d2_concat, name='d2_relu')

            # D3: (10, 7) -> (20, 13)
            # Stage1: (10, 7) -> (20, 14)
            d3_deconv = tf_utils.deconv2d(d2_relu, self.gen_c[11], name='d3_deconv2d')
            # Stage2: (20, 14) -> (20, 13)
            shapeA = e3_batchnorm.get_shape().as_list()[2]
            shapeB = d3_deconv.get_shape().as_list()[2] - e3_batchnorm.get_shape().as_list()[2]
            d3_split, _ = tf.split(d3_deconv, [shapeA, shapeB], axis=2, name='d3_split_2')
            tf_utils.print_activations(d3_split, logger=self.logger)
            # Stage3: Batch norm, concatenation, and relu
            d3_batchnorm = tf_utils.norm(d3_deconv, _type=self.norm, _ops=self._ops, logger=self.logger, name='d3_norm')
            d3_concat = tf.concat([d3_batchnorm, e3_batchnorm], axis=3, name='d3_concat')
            d3_relu = tf_utils.relu(d3_concat, name='d3_relu')

            # D4: (20, 13) -> (40, 25)
            # Stage1: (20, 13) -> (40, 26)
            d4_deconv = tf_utils.deconv2d(d3_relu, self.gen_c[12], name='d4_deconv2d')
            # Stage2: (40, 26) -> (40, 25)
            shapeA = e2_batchnorm.get_shape().as_list()[2]
            shapeB = d4_deconv.get_shape().as_list()[2] - e2_batchnorm.get_shape().as_list()[2]
            d4_split, _ = tf.split(d4_deconv, [shapeA, shapeB], axis=2, name='d4_split')
            tf_utils.print_activations(d4_split, logger=self.logger)
            # Stage3: Batch norm, concatenation, and relu
            d4_batchnorm = tf_utils.norm(d4_split, _type=self.norm, _ops=self._ops, logger=self.logger, name='d4_norm')
            d4_concat = tf.concat([d4_batchnorm, e2_batchnorm], axis=3, name='d4_concat')
            d4_relu = tf_utils.relu(d4_concat, name='d4_relu')

            # D5: (40, 25, 256) -> (80, 50, 128)
            d5_deconv = tf_utils.deconv2d(d4_relu, self.gen_c[13], name='d5_deconv2d')
            d5_batchnorm = tf_utils.norm(d5_deconv, _type=self.norm, _ops=self._ops, logger=self.logger, name='d5_norm')
            d5_concat = tf.concat([d5_batchnorm, e1_batchnorm], axis=3, name='d5_concat')
            d5_relu = tf_utils.relu(d5_concat, name='d5_relu')

            # D6: (80, 50, 128) -> (160, 100, 64)
            d6_deconv = tf_utils.deconv2d(d5_relu, self.gen_c[14], name='d6_deconv2d')
            d6_batchnorm = tf_utils.norm(d6_deconv, _type=self.norm, _ops=self._ops, logger=self.logger, name='d6_norm')
            d6_concat = tf.concat([d6_batchnorm, e0_conv2d], axis=3, name='d6_concat')
            d6_relu = tf_utils.relu(d6_concat, name='d6_relu')

            # D7: (160, 100, 64) -> (320, 200, 1)
            d7_deconv = tf_utils.deconv2d(d6_relu, self.gen_c[15], name='d7_deconv2d')
            output = tf_utils.tanh(d7_deconv, name='output_tanh')

            # Set reuse=True for next call
            self.reuse = True
            self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output


class Discriminator(object):
    def __init__(self, name=None, dis_c=None, norm='instance', logger=None, _ops=None):
        self.name = name
        self.dis_c = dis_c
        self.norm = norm
        self.logger = logger
        self._ops = _ops
        self.reuse = False

    def __call__(self, x):
        with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
            tf_utils.print_activations(x, logger=self.logger)

            # H1: (320, 200) -> (160, 100)
            h0_conv2d = tf_utils.conv2d(x, self.dis_c[0], name='h0_conv2d')
            h0_lrelu = tf_utils.lrelu(h0_conv2d, name='h0_lrelu')

            # H2: (160, 100) -> (80, 50)
            h1_conv2d = tf_utils.conv2d(h0_lrelu, self.dis_c[1], name='h1_conv2d')
            h1_norm = tf_utils.norm(h1_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h1_norm')
            h1_lrelu = tf_utils.lrelu(h1_norm, name='h1_lrelu')

            # H3: (80, 50) -> (40, 25)
            h2_conv2d = tf_utils.conv2d(h1_lrelu, self.dis_c[2], name='h2_conv2d')
            h2_norm = tf_utils.norm(h2_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h2_norm')
            h2_lrelu = tf_utils.lrelu(h2_norm, name='h2_lrelu')

            # H4: (40, 25) -> (20, 13)
            h3_conv2d = tf_utils.conv2d(h2_lrelu, self.dis_c[3], name='h3_conv2d')
            h3_norm = tf_utils.norm(h3_conv2d, _type=self.norm, _ops=self._ops, logger=self.logger, name='h3_norm')
            h3_lrelu = tf_utils.lrelu(h3_norm, name='h3_lrelu')

            # H5: (20, 13) -> (20, 13)
            output = tf_utils.conv2d(h3_lrelu, self.dis_c[4], k_h=3, k_w=3, d_h=1, d_w=1, name='output_conv2d')

            # set reuse=True for next call
            self.reuse = True
            self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output



