# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import logging
import tensorflow as tf

import utils as utils
import tensorflow_utils as tf_utils
from reader import Reader


class UNet(object):
    def __init__(self, decodeImgShape=(320, 400, 1), outputShape=(320, 200, 1), numClasses=4,
                 dataPath=(None, None), batchSize=1, lr=1e-3, weightDecay=1e-4, totalIters=2e5, isTrain=True,
                 logDir=None, name='UNet'):
        self.decodeImgShape = decodeImgShape
        self.inputShape = outputShape
        self.outputShape = outputShape
        self.numClasses = numClasses
        self.conv_dims = [64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024,
                          512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64, self.numClasses]

        self.dataPath = dataPath
        self.batchSize = batchSize
        self.lr = lr
        self.weightDecay = weightDecay
        self.totalSteps = totalIters
        self.startDecayStep = int(self.totalSteps * 0.5)
        self.decaySteps = self.totalSteps - self.startDecayStep
        self.isTrain = isTrain
        self.logDir = logDir
        self.name=name

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, logDir=self.logDir, isTrain=self.isTrain, name=self.name)

        self._build_graph()

    def _build_graph(self):
        # Input placeholders
        self.inputImgPh = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.inputShape], name='inputPh')
        self.keepProbPh = tf.compat.v1.placeholder(tf.float32, name='keepProbPh')
        trainReader = Reader(tfrecordsFile=self.dataPath[0],
                             decodeImgShape=self.decodeImgShape,
                             imgShape=self.inputShape,
                             batchSize=self.batchSize,
                             isTrain=self.isTrain)

        self.img, self.target = trainReader.shuffle_batch()

        self.predicion = self.forward(inputImg=self.img)

    def _init_tensorboard(self):
        print("Hello _init_tensoboard!")


    def forward(self, inputImg, padding='SAME'):
        # Stage 1
        tf_utils.print_activations(inputImg, logger=self.logger)
        s1_conv1 = tf_utils.conv2d(x=inputImg, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s1_conv1', logger=self.logger)
        s1_conv1 = tf_utils.relu(s1_conv1, name='relu_s1_conv1', logger=self.logger)
        s1_conv2 = tf_utils.conv2d(x=s1_conv1, output_dim=self.conv_dims[1], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s1_conv2', logger=self.logger)
        s1_conv2 = tf_utils.relu(s1_conv2, name='relu_s1_conv2', logger=self.logger)

        # Stage 2
        s2_maxpool = tf_utils.max_pool(x=s1_conv2, name='s2_maxpool2d', logger=self.logger)
        s2_conv1 = tf_utils.conv2d(x=s2_maxpool, output_dim=self.conv_dims[2], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s2_conv1', logger=self.logger)
        s2_conv1 = tf_utils.relu(s2_conv1, name='relu_s2_conv1', logger=self.logger)
        s2_conv2 = tf_utils.conv2d(x=s2_conv1, output_dim=self.conv_dims[3], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s2_conv2', logger=self.logger)
        s2_conv2 = tf_utils.relu(s2_conv2, name='relu_s2_conv2', logger=self.logger)

        # Stage 3
        s3_maxpool = tf_utils.max_pool(x=s2_conv2, name='s3_maxpool2d', logger=self.logger)
        s3_conv1 = tf_utils.conv2d(x=s3_maxpool, output_dim=self.conv_dims[4], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s3_conv1', logger=self.logger)
        s3_conv1 = tf_utils.relu(s3_conv1, name='relu_s3_conv1', logger=self.logger)
        s3_conv2 = tf_utils.conv2d(x=s3_conv1, output_dim=self.conv_dims[5], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s3_conv2', logger=self.logger)
        s3_conv2 = tf_utils.relu(s3_conv2, name='relu_s3_conv2', logger=self.logger)

        # Stage 4
        s4_maxpool = tf_utils.max_pool(x=s3_conv2, name='s4_maxpool2d', logger=self.logger)
        s4_conv1 = tf_utils.conv2d(x=s4_maxpool, output_dim=self.conv_dims[6], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s4_conv1', logger=self.logger)
        s4_conv1 = tf_utils.relu(s4_conv1, name='relu_s4_conv1', logger=self.logger)
        s4_conv2 = tf_utils.conv2d(x=s4_conv1, output_dim=self.conv_dims[7], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s4_conv2', logger=self.logger)
        s4_conv2 = tf_utils.relu(s4_conv2, name='relu_s4_conv2', logger=self.logger)
        s4_conv2_drop = tf_utils.dropout(x=s4_conv2, keep_prob=self.keepProbPh, name='s4_dropout',
                                         logger=self.logger)

        # Stage 5
        s5_maxpool = tf_utils.max_pool(x=s4_conv2_drop, name='s5_maxpool2d', logger=self.logger)
        s5_conv1 = tf_utils.conv2d(x=s5_maxpool, output_dim=self.conv_dims[8], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s5_conv1', logger=self.logger)
        s5_conv1 = tf_utils.relu(s5_conv1, name='relu_s5_conv1', logger=self.logger)
        s5_conv2 = tf_utils.conv2d(x=s5_conv1, output_dim=self.conv_dims[9], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s5_conv2', logger=self.logger)
        s5_conv2 = tf_utils.relu(s5_conv2, name='relu_s5_conv2', logger=self.logger)
        s5_conv2_drop = tf_utils.dropout(x=s5_conv2, keep_prob=self.keepProbPh, name='s5_dropout',
                                         logger=self.logger)

        # Stage 6
        s6_deconv1 = tf_utils.deconv2d(x=s5_conv2_drop, output_dim=self.conv_dims[10], k_h=2, k_w=2, initializer='He',
                                       name='s6_deconv1', logger=self.logger)
        s6_deconv1 = tf_utils.relu(s6_deconv1, name='relu_s6_deconv1', logger=self.logger)
        # Cropping
        w1 = s4_conv2_drop.get_shape().as_list()[2]
        w2 = s6_deconv1.get_shape().as_list()[2] - s4_conv2_drop.get_shape().as_list()[2]
        s6_deconv1_split, _ = tf.split(s6_deconv1, num_or_size_splits=[w1, w2], axis=2, name='axis2_split')
        tf_utils.print_activations(s6_deconv1_split, logger=self.logger)
        # Concat
        s6_concat = tf_utils.concat(values=[s6_deconv1_split, s4_conv2_drop], axis=3, name='s6_axis3_concat',
                                    logger=self.logger)
        s6_conv2 = tf_utils.conv2d(x=s6_concat, output_dim=self.conv_dims[11], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s6_conv2', logger=self.logger)
        s6_conv2 = tf_utils.relu(s6_conv2, name='relu_s6_conv2', logger=self.logger)
        s6_conv3 = tf_utils.conv2d(x=s6_conv2, output_dim=self.conv_dims[12], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s6_conv3', logger=self.logger)
        s6_conv3 = tf_utils.relu(s6_conv3, name='relu_s6_conv3', logger=self.logger)

        # Stage 7
        s7_deconv1 = tf_utils.deconv2d(x=s6_conv3, output_dim=self.conv_dims[13], k_h=2, k_w=2, initializer='He',
                                       name='s7_deconv1', logger=self.logger)
        s7_deconv1 = tf_utils.relu(s7_deconv1, name='relu_s7_deconv1', logger=self.logger)
        # Concat
        s7_concat = tf_utils.concat(values=[s7_deconv1, s3_conv2], axis=3, name='s7_axis3_concat', logger=self.logger)
        s7_conv2 = tf_utils.conv2d(x=s7_concat, output_dim=self.conv_dims[14], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s7_conv2', logger=self.logger)
        s7_conv2 = tf_utils.relu(s7_conv2, name='relu_s7_conv2', logger=self.logger)
        s7_conv3 = tf_utils.conv2d(x=s7_conv2, output_dim=self.conv_dims[15], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s7_conv3', logger=self.logger)
        s7_conv3 = tf_utils.relu(s7_conv3, name='relu_s7_conv3', logger=self.logger)

        # Stage 8
        s8_deconv1 = tf_utils.deconv2d(x=s7_conv3, output_dim=self.conv_dims[16], k_h=2, k_w=2, initializer='He',
                                       name='s8_deconv1', logger=self.logger)
        s8_deconv1 = tf_utils.relu(s8_deconv1, name='relu_s8_deconv1', logger=self.logger)
        # Concat
        s8_concat = tf_utils.concat(values=[s8_deconv1,s2_conv2], axis=3, name='s8_axis3_concat', logger=self.logger)
        s8_conv2 = tf_utils.conv2d(x=s8_concat, output_dim=self.conv_dims[17], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s8_conv2', logger=self.logger)
        s8_conv2 = tf_utils.relu(s8_conv2, name='relu_s8_conv2', logger=self.logger)
        s8_conv3 = tf_utils.conv2d(x=s8_conv2, output_dim=self.conv_dims[18], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s8_conv3', logger=self.logger)
        s8_conv3 = tf_utils.relu(s8_conv3, name='relu_conv3', logger=self.logger)

        # Stage 9
        s9_deconv1 = tf_utils.deconv2d(x=s8_conv3, output_dim=self.conv_dims[19], k_h=2, k_w=2, initializer='He',
                                       name='s9_deconv1', logger=self.logger)
        s9_deconv1 = tf_utils.relu(s9_deconv1, name='relu_s9_deconv1', logger=self.logger)
        # Concat
        s9_concat = tf_utils.concat(values=[s9_deconv1, s1_conv2], axis=3, name='s9_axis3_concat', logger=self.logger)
        s9_conv2 = tf_utils.conv2d(x=s9_concat, output_dim=self.conv_dims[20], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s9_conv2', logger=self.logger)
        s9_conv2 = tf_utils.relu(s9_conv2, name='relu_s9_conv2', logger=self.logger)
        s9_conv3 = tf_utils.conv2d(x=s9_conv2, output_dim=self.conv_dims[21], k_h=3, k_w=3, d_h=1, d_w=1,
                                   padding=padding, initializer='He', name='s9_conv3', logger=self.logger)
        s9_conv3 = tf_utils.relu(s9_conv3, name='relu_s9_conv3', logger=self.logger)

        output = tf_utils.conv2d(s9_conv3, output_dim=self.conv_dims[22], k_h=1, k_w=1, d_h=1, d_w=1, padding=padding,
                                 initializer='He', name='output', logger=self.logger)
        return output

