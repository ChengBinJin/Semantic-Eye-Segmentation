# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import logging
import math
import numpy as np
import tensorflow as tf

import utils as utils
import tensorflow_utils as tf_utils
from reader import Reader

class DenseUNet(object):
    def __init__(self, decodeImgShape=(320, 400, 1), outputShape=(320, 200, 1), numClasses=4,
                 dataPath=(None, None), batchSize=1, lr=1e-3, weightDecay=1e-4, totalIters=2e5, isTrain=True,
                 logDir=None, method=None, multi_test=True, resize_factor=0.5, use_dice_loss=False,
                 use_batch_norm=False, lambda_one=1.0, name='UNet'):
        self.decodeImgShape = decodeImgShape
        self.inputShape = outputShape
        self.outputShape = outputShape
        self.numClasses = numClasses
        self.method = method
        self.use_batch_norm = use_batch_norm
        self.isTrain = isTrain
        self.resize_factor = resize_factor
        self.use_dice_loss = use_dice_loss
        self.lambda_one = lambda_one

        self.multi_test = False if self.isTrain else multi_test
        self.degree = 10
        self.num_try = len(range(-self.degree, self.degree+1, 2))  # multi_tes: from -10 degree to 11 degrees
        self.conv_dims = [16, 32, 32, 32, 32, 32, 32, 32, 64, 64,
                         32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16]

        self.dataPath = dataPath
        self.batchSize = batchSize
        self.lr = lr
        self.weightDecay = weightDecay
        self.totalSteps = totalIters
        self.startDecayStep = int(self.totalSteps * 0.5)
        self.decaySteps = self.totalSteps - self.startDecayStep
        self.logDir = logDir
        self.name=name

        self.mIoUMetric, self.mIoUMetricUpdate = None, None
        self.tb_lr = None
        self._ops = list()

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)
        utils.init_logger(logger=self.logger, logDir=self.logDir, isTrain=self.isTrain, name=self.name)

        self._build_graph()             # main graph
        self._init_eval_graph()         # evaluation for validation data
        self._init_test_graph()         # for test data
        self._best_metrics_record()     # metrics
        self._init_tensorboard()        # tensorboard
        tf_utils.show_all_variables(logger=self.logger if self.isTrain else None)

    def _build_graph(self):
        # Input placeholders
        self.inputImgPh = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.inputShape], name='inputPh')
        self.ratePh = tf.compat.v1.placeholder(tf.float32, name='keepProbPh')
        self.trainMode = tf.compat.v1.placeholder(tf.bool, name='train_mode_ph')

        # Initialize TFRecoder reader
        trainReader = Reader(tfrecordsFile=self.dataPath[0],
                             decodeImgShape=self.decodeImgShape,
                             imgShape=self.inputShape,
                             batchSize=self.batchSize,
                             name='train')

        # Random batch for training
        self.imgTrain, self.segImgTrain = trainReader.shuffle_batch()

        # Network forward for training
        self.predTrain = self.forward_network(inputImg=self.normalize(self.imgTrain), reuse=False)
        self.predClsTrain = tf.math.argmax(self.predTrain, axis=-1)

        # Data loss
        self.dataLoss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.predTrain,
            labels=self.convert_one_hot(self.segImgTrain)))

        # Regularization term
        variables = self.get_regularization_variables()
        self.regTerm = self.weightDecay * tf.math.reduce_mean([tf.nn.l2_loss(variable) for variable in variables])

        # Additional loss function
        # Dice coefficient loss term
        self.dice_loss = tf.constant(0.)
        if self.use_dice_loss:
            self.dice_loss = self.generalized_dice_loss(labels=self.segImgTrain, logits=self.predTrain,
                                                        hyper_parameter=self.lambda_one)

        # Total loss = Data loss + Regularization term + Dice coefficient loss
        self.totalLoss = self.dataLoss + self.regTerm + self.dice_loss

        # Optimizer
        train_op = self.init_optimizer(loss=self.totalLoss, name='Adam')
        train_ops = [train_op] + self._ops
        self.trainOp = tf.group(*train_ops)

    def generalized_dice_loss(self, labels, logits, hyper_parameter=1.0):
        # This implementation refers to srcolinas's dice_loss.py
        # (https://gist.github.com/srcolinas/6df2e5e21c11227a04f826322081addf)

        smooth = 1e-17
        labels = self.convert_one_hot(labels)
        logits = tf.nn.softmax(logits)

        # weights = 1.0 / (tf.reduce_sum(labels, axis=[0, 1, 2])**2)
        weights = tf.math.divide(1.0, (tf.math.square(tf.math.reduce_sum(labels, axis=[0, 1, 2])) + smooth))

        # Numerator part
        numerator = tf.math.reduce_sum(labels * logits, axis=[0, 1, 2])
        numerator = tf.reduce_sum(weights * numerator)

        # Denominator part
        denominator = tf.math.reduce_sum(labels + logits, axis=[0, 1, 2])
        denominator = tf.math.reduce_sum(weights * denominator)

        # Dice coeeficient loss
        loss = hyper_parameter * (1.0 - 2.0 * (numerator + smooth) / (denominator + smooth))

        return loss

    def _init_eval_graph(self):
        # Initialize TFRecoder reader
        valReader = Reader(tfrecordsFile=self.dataPath[1],
                           decodeImgShape=self.decodeImgShape,
                           imgShape=self.inputShape,
                           batchSize=1,
                           name='validation')

        # Batch for validation data
        imgVal, segImgVal, self.img_name_val, self.user_id_val = valReader.batch(
            multi_test= False if self.isTrain else self.multi_test)

        # tf.train.batch() returns [None, H, M, D]
        # For tf.metrics.mean_iou we need [batch_size, H, M, D]
        if self.multi_test:
            shape = [self.num_try, *self.outputShape]
        else:
            shape = [1, *self.outputShape]

        # Roateds img and segImg are step 1
        imgVal = tf.reshape(imgVal, shape=shape)
        segImgVal = tf.reshape(segImgVal, shape=shape)

        # Network forward for validation data
        predVal = self.forward_network(inputImg=self.normalize(imgVal), reuse=True)

        # Since multi_test, we need inversely rotate back to the original segImg
        if self.multi_test:
            # Step 1: original rotated images
            self.imgVal_s1, self.predVal_s1, self.segImgVal_s1 = imgVal, predVal, segImgVal

            # Step 2: inverse-rotated images
            self.imgVal_s2, self.predVal_s2, self.segImgVal_s2 = self.roate_independently(
                imgVal, predVal, segImgVal)

            # Step 3: combine all results to estimate the final result
            sum_all = tf.math.reduce_sum(self.predVal_s2, axis=0)   # [N, H, W, num_actions] -> [H, W, num_actions]
            sum_all = tf.expand_dims(sum_all, axis=0)               # [H, W, num_actions] -> [1, H, W, num_actions]
            predVal_s3 = tf.math.argmax(sum_all, axis=-1)           # [1, H, W]

            _, h, w, c = imgVal.get_shape().as_list()
            base_id = int(np.floor(self.num_try / 2.))
            self.imgVal = tf.slice(imgVal, begin=[base_id, 0, 0, 0], size=[1, h, w, c])
            self.segImgVal = tf.slice(segImgVal, begin=[base_id, 0, 0, 0], size=[1, h, w, c])
            self.predClsVal = predVal_s3
        else:
            self.imgVal = imgVal
            self.segImgVal = segImgVal
            self.predClsVal = tf.math.argmax(predVal, axis=-1)

        with tf.compat.v1.name_scope('Metrics'):
            # Calculate mean IoU using TensorFlow
            self.mIoU_metric, self.mIoU_metric_update = tf.compat.v1.metrics.mean_iou(
                labels=tf.squeeze(self.segImgVal, axis=-1),
                predictions=self.predClsVal,
                num_classes=self.numClasses)

            # Calculate accuracy using TensorFlow
            self.accuracy_metric, self.accuracy_metric_update = tf.compat.v1.metrics.accuracy(
                labels=tf.squeeze(self.segImgVal, axis=-1),
                predictions=self.predClsVal)

            # Calculate precision using TensorFlow
            self.precision_metric, self.precision_metric_update = tf.compat.v1.metrics.precision(
                labels=tf.squeeze(self.segImgVal, axis=-1),
                predictions=self.predClsVal)

            # Calculate recall using TensorFlow
            self.recall_metric, self.recall_metric_update = tf.compat.v1.metrics.recall(
                labels=tf.squeeze(self.segImgVal, axis=-1),
                predictions=self.predClsVal)

            # Calculate F1 score
            self.f1_score_metric = tf.math.divide(2 * self.precision_metric * self.recall_metric ,
                                            (self.precision_metric + self.recall_metric))

            # Calculate per-class accuracy
            _, self.per_class_accuracy_metric_update = tf.compat.v1.metrics.mean_per_class_accuracy(
                    labels=tf.squeeze(self.segImgVal),
                    predictions=self.predClsVal,
                    num_classes=self.numClasses)

        # Isolate the variables stored behind the scens by the metric operation
        running_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope='Metrics')

        # Define initializer to initialie/reset running variables
        self.running_vars_initializer = tf.compat.v1.variables_initializer(var_list=running_vars)

    def roate_independently(self, imgVal, predVal, segImgVal=None, is_test=False):
        imgs, preds = list(), list()
        segImgs, segImg = None, None
        if not is_test:
            segImgs = list()

        for idx, degree in enumerate(range(self.degree, -self.degree - 1, -2)):
            n, h, w, c = imgVal.get_shape().as_list()

            # Extract spectific tensor
            img = tf.slice(imgVal, begin=[idx, 0, 0, 0], size=[1, h, w, c])                     # [1, H, W, 1]
            pred = tf.slice(predVal, begin=[idx, 0, 0, 0], size=[1, h, w, self.numClasses])     # [1, H, W, num_classes]
            if not is_test:
                segImg = tf.slice(segImgVal, begin=[idx, 0, 0, 0], size=[1, h, w, c])           # [1, H, W, 1]

            # From degree to radian
            radian = degree * math.pi / 180.

            # Roate img and segImgs
            imgs.append(tf.contrib.image.rotate(images=img, angles=radian, interpolation='BILINEAR'))
            preds.append(tf.contrib.image.rotate(images=pred, angles=radian, interpolation='BILINEAR'))
            if not is_test:
                segImgs.append(tf.contrib.image.rotate(images=segImg, angles=radian, interpolation='NEAREST'))

        if not is_test:
            return tf.concat(imgs, axis=0), tf.concat(preds, axis=0), tf.concat(segImgs, axis=0)
        else:
            return tf.concat(imgs, axis=0), tf.concat(preds, axis=0)

    def _init_test_graph(self):
        # Initialize TFRecoder reader
        testReader = Reader(tfrecordsFile=self.dataPath[0],
                            decodeImgShape=self.decodeImgShape,
                            imgShape=self.inputShape,
                            batchSize=1,
                            name='test')

        # Batch for test data
        imgTest, _, self.img_name_test, self.user_id_test = testReader.batch(multi_test=self.multi_test)

        # Convert the shape [?, self.num_try, H, W, 1] to [self.num_try, H, W, 1] for multi-test
        if self.multi_test:
            shape = [self.num_try, *self.outputShape]
        else:
            shape = [1, *self.outputShape]
        imgTest = tf.reshape(imgTest, shape=shape)

        # Network forward for test data
        predTest = self.forward_network(inputImg=self.normalize(imgTest), reuse=True)

        # Since multi_test, we need inversely rotate back to the original segImg
        if self.multi_test:
            # Step 1: original rotated images
            self.imgTest_s1, self.predTest_s1 = imgTest, predTest

            # Step 2: inverse-rotated images
            self.imgTest_s2, self.predTest_s2= self.roate_independently(self.imgTest_s1, self.predTest_s1, is_test=True)

            # Step 3: combine all results to estimate the final result
            sum_all = tf.math.reduce_sum(self.predTest_s2, axis=0)   # [N, H, W, num_actions] -> [H, W, num_actions]
            sum_all = tf.expand_dims(sum_all, axis=0)                # [H, W, num_actions] -> [1, H, W, num_actions]
            predTest_s3 = tf.math.argmax(sum_all, axis=-1)           # [1, H, W]

            _, h, w, c = imgTest.get_shape().as_list()
            base_id = int(np.floor(self.num_try / 2.))
            self.imgTest = tf.slice(imgTest, begin=[base_id, 0, 0, 0], size=[1, h, w, c])
            self.predClsTest = predTest_s3
        else:
            self.imgTest = imgTest
            self.predClsTest = tf.math.argmax(predTest, axis=-1)

    def _best_metrics_record(self):
        self.best_mIoU_ph = tf.compat.v1.placeholder(tf.float32, name='best_mIoU')
        self.best_acc_ph = tf.compat.v1.placeholder(tf.float32, name='best_acc')
        self.best_precision_ph = tf.compat.v1.placeholder(tf.float32, name='best_precision')
        self.best_recall_ph = tf.compat.v1.placeholder(tf.float32, name='best_recall')
        self.best_f1_score_ph = tf.compat.v1.placeholder(tf.float32, name='best_f1_score')

        # Best mIoU variable
        self.best_mIoU = tf.compat.v1.get_variable(name='best_mIoU', dtype=tf.float32, initializer=tf.constant(0.),
                                                   trainable=False)
        self.assign_best_mIoU = tf.assign(self.best_mIoU, value=self.best_mIoU_ph)

        # Best acciracy variable
        self.best_acc = tf.compat.v1.get_variable(name='best_acc', dtype=tf.float32, initializer=tf.constant(0.),
                                                  trainable=False)
        self.assign_best_acc = tf.assign(self.best_acc, value=self.best_acc_ph)

        # Best precision variable
        self.best_precision = tf.compat.v1.get_variable(name='best_precision', dtype=tf.float32, initializer=tf.constant(0.),
                                                       trainable=False)
        self.assign_best_precision = tf.assign(self.best_precision, value=self.best_precision_ph)

        # Best recall variable
        self.best_recall = tf.compat.v1.get_variable(name='best_recall', dtype=tf.float32, initializer=tf.constant(0.),
                                                     trainable=False)
        self.assign_best_recall = tf.assign(self.best_recall, value=self.best_recall_ph)

        # # Best f1_score variable
        self.best_f1_score = tf.compat.v1.get_variable(name='best_f1_score', dtype=tf.float32, initializer=tf.constant(0.),
                                                       trainable=False)
        self.assign_best_f1_score = tf.assign(self.best_f1_score, value=self.best_f1_score_ph)

    def init_optimizer(self, loss, name=None):
        with tf.compat.v1.variable_scope(name):
            globalStep = tf.Variable(0., dtype=tf.float32, trainable=False)
            startLearningRate = self.lr
            endLearningRate = self.lr * 0.001
            startDecayStep = self.startDecayStep
            decaySteps = self.decaySteps

            learningRate = (tf.where(tf.greater_equal(globalStep, startDecayStep),
                                     tf.compat.v1.train.polynomial_decay(startLearningRate,
                                                               globalStep - startDecayStep,
                                                               decaySteps, endLearningRate, power=1.0),
                                     startLearningRate))
            self.tb_lr = tf.compat.v1.summary.scalar('learning_rate', learningRate)

            learnStep = tf.compat.v1.train.AdamOptimizer(learning_rate=learningRate, beta1=0.99).minimize(
                loss, global_step=globalStep)

        return learnStep

    def _init_tensorboard(self):
        self.tb_total = tf.summary.scalar('Loss/total_loss', self.totalLoss)
        self.tb_data = tf.summary.scalar('Loss/data_loss', self.dataLoss)
        self.tb_reg = tf.summary.scalar('Loss/reg_term', self.regTerm)
        self.tb_dice = tf.summary.scalar('Loss/dice_loss', self.dice_loss)
        self.summary_op = tf.summary.merge(inputs=[self.tb_total, self.tb_data, self.tb_reg, self.tb_dice, self.tb_lr])

        self.tb_mIoU = tf.summary.scalar('Acc/mIoU', self.mIoU_metric)
        self.tb_accuracy = tf.summary.scalar('Acc/accuracy', self.accuracy_metric)
        self.tb_precision = tf.summary.scalar('Acc/precision', self.precision_metric)
        self.tb_recall = tf.summary.scalar('Acc/recall', self.recall_metric)

        self.tb_f1_score = tf.summary.scalar('Acc/f1_score', self.f1_score_metric)
        self.metric_summary_op = tf.summary.merge(inputs=[self.tb_mIoU, self.tb_accuracy,
                                                          self.tb_precision, self.tb_recall, self.tb_f1_score])

    @staticmethod
    def get_regularization_variables():
        # We exclude 'bias', 'beta' and 'gamma' in batch normalization
        variables = [variable for variable in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
                     if ('bias' not in variable.name) and
                     ('beta' not in variable.name) and
                     ('gamma' not in variable.name)]

        return variables

    @staticmethod
    def normalize(data):
        return data / 127.5 - 1.0

    def convert_one_hot(self, data):
        shape = data.get_shape().as_list()
        data = tf.dtypes.cast(data, dtype=tf.uint8)
        data = tf.one_hot(data, depth=self.numClasses, axis=-1, dtype=tf.float32, name='one_hot')
        data = tf.reshape(data, shape=[*shape[:3], self.numClasses])

        return data

    def forward_network(self, inputImg, padding='SAME', reuse=False):
        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            # This part is for compatible between input size [640, 400] and [320, 200]
            if self.resize_factor == 1.0:
                # Stage 0
                tf_utils.print_activations(inputImg, logger=self.logger)
                s0_conv1 = tf_utils.conv2d(x=inputImg, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=padding, initializer='He', name='s0_conv1', logger=self.logger)
                s0_conv1 = tf_utils.relu(s0_conv1, name='relu_s0_conv1', logger=self.logger)

                s0_conv2 = tf_utils.conv2d(x=s0_conv1, output_dim=2*self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=padding, initializer='He', name='s0_conv2', logger=self.logger)
                if self.use_batch_norm:
                    s0_conv2 = tf_utils.norm(s0_conv2, name='s0_norm1', _type='batch', _ops=self._ops,
                                             is_train=self.trainMode, logger=self.logger)
                s0_conv2 = tf_utils.relu(s0_conv2, name='relu_s0_conv2', logger=self.logger)

                # Stage 1
                s1_maxpool = tf_utils.max_pool(x=s0_conv2, name='s1_maxpool2d', logger=self.logger)

                s1_conv1 = tf_utils.conv2d(x=s1_maxpool, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=padding, initializer='He', name='s1_conv1', logger=self.logger)
                if self.use_batch_norm:
                    s1_conv1 = tf_utils.norm(s1_conv1, name='s1_norm0', _type='batch', _ops=self._ops,
                                             is_train=self.trainMode, logger=self.logger)
                s1_conv1 = tf_utils.relu(s1_conv1, name='relu_s1_conv1', logger=self.logger)

                s1_conv2 = tf_utils.conv2d(x=s1_conv1, output_dim=self.conv_dims[1], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=padding, initializer='He', name='s1_conv2', logger=self.logger)
                if self.use_batch_norm:
                    s1_conv2 = tf_utils.norm(s1_conv2, name='s1_norm1', _type='batch', _ops=self._ops,
                                             is_train=self.trainMode, logger=self.logger)
                s1_conv2 = tf_utils.relu(s1_conv2, name='relu_s1_conv2', logger=self.logger)
            else:
                # Stage 1
                tf_utils.print_activations(inputImg, logger=self.logger)
                s1_conv1 = tf_utils.conv2d(x=inputImg, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=padding, initializer='He', name='s1_conv1', logger=self.logger)
                s1_conv1 = tf_utils.relu(s1_conv1, name='relu_s1_conv1', logger=self.logger)

                s1_conv2 = tf_utils.conv2d(x=s1_conv1, output_dim=self.conv_dims[1], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=padding, initializer='He', name='s1_conv2', logger=self.logger)
                if self.use_batch_norm:
                    s1_conv2 = tf_utils.norm(s1_conv2, name='s1_norm1', _type='batch', _ops=self._ops,
                                             is_train=self.trainMode, logger=self.logger)
                s1_conv2 = tf_utils.relu(s1_conv2, name='relu_s1_conv2', logger=self.logger)

            # Stage 2
            s2_maxpool = tf_utils.max_pool(x=s1_conv2, name='s2_maxpool2d', logger=self.logger)
            s2_conv1 = tf_utils.conv2d(x=s2_maxpool, output_dim=self.conv_dims[2], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s2_conv1', logger=self.logger)
            if self.use_batch_norm:
                s2_conv1 = tf_utils.norm(s2_conv1, name='s2_norm0', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s2_conv1 = tf_utils.relu(s2_conv1, name='relu_s2_conv1', logger=self.logger)

            s2_conv2 = tf_utils.conv2d(x=s2_conv1, output_dim=self.conv_dims[3], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s2_conv2', logger=self.logger)
            if self.use_batch_norm:
                s2_conv2 = tf_utils.norm(s2_conv2, name='s2_norm1', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s2_conv2 = tf_utils.relu(s2_conv2, name='relu_s2_conv2', logger=self.logger)

            # Stage 3
            s3_maxpool = tf_utils.max_pool(x=s2_conv2, name='s3_maxpool2d', logger=self.logger)
            s3_conv1 = tf_utils.conv2d(x=s3_maxpool, output_dim=self.conv_dims[4], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s3_conv1', logger=self.logger)
            if self.use_batch_norm:
                s3_conv1 = tf_utils.norm(s3_conv1, name='s3_norm0', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s3_conv1 = tf_utils.relu(s3_conv1, name='relu_s3_conv1', logger=self.logger)

            s3_conv2 = tf_utils.conv2d(x=s3_conv1, output_dim=self.conv_dims[5], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s3_conv2', logger=self.logger)
            if self.use_batch_norm:
                s3_conv2 = tf_utils.norm(s3_conv2, name='s3_norm1', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s3_conv2 = tf_utils.relu(s3_conv2, name='relu_s3_conv2', logger=self.logger)

            # Stage 4
            s4_maxpool = tf_utils.max_pool(x=s3_conv2, name='s4_maxpool2d', logger=self.logger)
            s4_conv1 = tf_utils.conv2d(x=s4_maxpool, output_dim=self.conv_dims[6], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s4_conv1', logger=self.logger)
            if self.use_batch_norm:
                s4_conv1 = tf_utils.norm(s4_conv1, name='s4_norm0', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s4_conv1 = tf_utils.relu(s4_conv1, name='relu_s4_conv1', logger=self.logger)

            s4_conv2 = tf_utils.conv2d(x=s4_conv1, output_dim=self.conv_dims[7], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s4_conv2', logger=self.logger)
            if self.use_batch_norm:
                s4_conv2 = tf_utils.norm(s4_conv2, name='s4_norm1', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s4_conv2 = tf_utils.relu(s4_conv2, name='relu_s4_conv2', logger=self.logger)
            s4_conv2_drop = tf_utils.dropout(x=s4_conv2, keep_prob=self.ratePh, name='s4_dropout',
                                             logger=self.logger)

            # Stage 5
            s5_maxpool = tf_utils.max_pool(x=s4_conv2_drop, name='s5_maxpool2d', logger=self.logger)
            s5_conv1 = tf_utils.conv2d(x=s5_maxpool, output_dim=self.conv_dims[8], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s5_conv1', logger=self.logger)
            if self.use_batch_norm:
                s5_conv1 = tf_utils.norm(s5_conv1, name='s5_norm0', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s5_conv1 = tf_utils.relu(s5_conv1, name='relu_s5_conv1', logger=self.logger)

            s5_conv2 = tf_utils.conv2d(x=s5_conv1, output_dim=self.conv_dims[9], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s5_conv2', logger=self.logger)
            if self.use_batch_norm:
                s5_conv2 = tf_utils.norm(s5_conv2, name='s5_norm1', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s5_conv2 = tf_utils.relu(s5_conv2, name='relu_s5_conv2', logger=self.logger)
            s5_conv2_drop = tf_utils.dropout(x=s5_conv2, keep_prob=self.ratePh, name='s5_dropout',
                                             logger=self.logger)

            # Stage 6
            s6_deconv1 = tf_utils.deconv2d(x=s5_conv2_drop, output_dim=self.conv_dims[10], k_h=2, k_w=2,
                                           initializer='He', name='s6_deconv1', logger=self.logger)
            if self.use_batch_norm:
                s6_deconv1 = tf_utils.norm(s6_deconv1, name='s6_norm0', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
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
            if self.use_batch_norm:
                s6_conv2 = tf_utils.norm(s6_conv2, name='s6_norm1', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s6_conv2 = tf_utils.relu(s6_conv2, name='relu_s6_conv2', logger=self.logger)

            # Addition
            s6_conv2 = tf_utils.identity(s6_conv2 + s4_conv1, name='stage6_add', logger=self.logger)

            s6_conv3 = tf_utils.conv2d(x=s6_conv2, output_dim=self.conv_dims[12], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s6_conv3', logger=self.logger)
            if self.use_batch_norm:
                s6_conv3 = tf_utils.norm(s6_conv3, name='s6_norm2', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s6_conv3 = tf_utils.relu(s6_conv3, name='relu_s6_conv3', logger=self.logger)

            # Stage 7
            s7_deconv1 = tf_utils.deconv2d(x=s6_conv3, output_dim=self.conv_dims[13], k_h=2, k_w=2, initializer='He',
                                           name='s7_deconv1', logger=self.logger)
            if self.use_batch_norm:
                s7_deconv1 = tf_utils.norm(s7_deconv1, name='s7_norm0', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s7_deconv1 = tf_utils.relu(s7_deconv1, name='relu_s7_deconv1', logger=self.logger)
            # Concat
            s7_concat = tf_utils.concat(values=[s7_deconv1, s3_conv2], axis=3, name='s7_axis3_concat',
                                        logger=self.logger)

            s7_conv2 = tf_utils.conv2d(x=s7_concat, output_dim=self.conv_dims[14], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s7_conv2', logger=self.logger)
            if self.use_batch_norm:
                s7_conv2 = tf_utils.norm(s7_conv2, name='s7_norm1', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s7_conv2 = tf_utils.relu(s7_conv2, name='relu_s7_conv2', logger=self.logger)

            # Addition
            s7_conv2 = tf_utils.identity(s7_conv2 + s3_conv1, name='stage7_add', logger=self.logger)

            s7_conv3 = tf_utils.conv2d(x=s7_conv2, output_dim=self.conv_dims[15], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s7_conv3', logger=self.logger)
            if self.use_batch_norm:
                s7_conv3 = tf_utils.norm(s7_conv3, name='s7_norm2', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s7_conv3 = tf_utils.relu(s7_conv3, name='relu_s7_conv3', logger=self.logger)

            # Stage 8
            s8_deconv1 = tf_utils.deconv2d(x=s7_conv3, output_dim=self.conv_dims[16], k_h=2, k_w=2, initializer='He',
                                           name='s8_deconv1', logger=self.logger)
            if self.use_batch_norm:
                s8_deconv1 = tf_utils.norm(s8_deconv1, name='s8_norm0', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s8_deconv1 = tf_utils.relu(s8_deconv1, name='relu_s8_deconv1', logger=self.logger)
            # Concat
            s8_concat = tf_utils.concat(values=[s8_deconv1,s2_conv2], axis=3, name='s8_axis3_concat',
                                        logger=self.logger)

            s8_conv2 = tf_utils.conv2d(x=s8_concat, output_dim=self.conv_dims[17], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s8_conv2', logger=self.logger)
            if self.use_batch_norm:
                s8_conv2 = tf_utils.norm(s8_conv2, name='s8_norm1', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s8_conv2 = tf_utils.relu(s8_conv2, name='relu_s8_conv2', logger=self.logger)

            # Addition
            s8_conv2 = tf_utils.identity(s8_conv2 + s2_conv1, name='stage8_add', logger=self.logger)

            s8_conv3 = tf_utils.conv2d(x=s8_conv2, output_dim=self.conv_dims[18], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s8_conv3', logger=self.logger)
            if self.use_batch_norm:
                s8_conv3 = tf_utils.norm(s8_conv3, name='s8_norm2', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s8_conv3 = tf_utils.relu(s8_conv3, name='relu_conv3', logger=self.logger)

            # Stage 9
            s9_deconv1 = tf_utils.deconv2d(x=s8_conv3, output_dim=self.conv_dims[19], k_h=2, k_w=2,
                                           initializer='He', name='s9_deconv1', logger=self.logger)
            if self.use_batch_norm:
                s9_deconv1 = tf_utils.norm(s9_deconv1, name='s9_norm0', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s9_deconv1 = tf_utils.relu(s9_deconv1, name='relu_s9_deconv1', logger=self.logger)
            # Concat
            s9_concat = tf_utils.concat(values=[s9_deconv1, s1_conv2], axis=3, name='s9_axis3_concat',
                                        logger=self.logger)

            s9_conv2 = tf_utils.conv2d(x=s9_concat, output_dim=self.conv_dims[20], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s9_conv2', logger=self.logger)
            if self.use_batch_norm:
                s9_conv2 = tf_utils.norm(s9_conv2, name='s9_norm1', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s9_conv2 = tf_utils.relu(s9_conv2, name='relu_s9_conv2', logger=self.logger)

            # Addition
            s9_conv2 = tf_utils.identity(s9_conv2 + s1_conv1, name='stage9_add', logger=self.logger)

            s9_conv3 = tf_utils.conv2d(x=s9_conv2, output_dim=self.conv_dims[21], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s9_conv3', logger=self.logger)
            if self.use_batch_norm:
                s9_conv3 = tf_utils.norm(s9_conv3, name='s9_norm2', _type='batch', _ops=self._ops,
                                         is_train=self.trainMode, logger=self.logger)
            s9_conv3 = tf_utils.relu(s9_conv3, name='relu_s9_conv3', logger=self.logger)

            if self.resize_factor == 1.0:
                s10_deconv1 = tf_utils.deconv2d(x=s9_conv3, output_dim=self.conv_dims[-1], k_h=2, k_w=2,
                                                initializer='He', name='s10_deconv1', logger=self.logger)
                if self.use_batch_norm:
                    s10_deconv1 = tf_utils.norm(s10_deconv1, name='s10_norm0', _type='batch', _ops=self._ops,
                                             is_train=self.trainMode, logger=self.logger)
                s10_deconv1 = tf_utils.relu(s10_deconv1, name='relu_s10_deconv1', logger=self.logger)
                # Concat
                s10_concat = tf_utils.concat(values=[s10_deconv1, s0_conv2], axis=3, name='s10_axis3_concat',
                                             logger=self.logger)

                s10_conv2 = tf_utils.conv2d(s10_concat, output_dim=self.conv_dims[-1], k_h=3, k_w=3, d_h=1, d_w=1,
                                            padding=padding, initializer='He', name='s10_conv2', logger=self.logger)
                if self.use_batch_norm:
                    s10_conv2 = tf_utils.norm(s10_conv2, name='s10_norm1', _type='batch', _ops=self._ops,
                                             is_train=self.trainMode, logger=self.logger)
                s10_conv2 = tf_utils.relu(s10_conv2, name='relu_s10_conv2', logger=self.logger)

                # Addition
                s10_conv2 = tf_utils.identity(s10_conv2 + s0_conv1, name='s10_add', logger=self.logger)

                s10_conv3 = tf_utils.conv2d(x=s10_conv2, output_dim=self.conv_dims[-1], k_h=3, k_w=3, d_h=1, d_w=1,
                                            padding=padding, initializer='He', name='s10_conv3', logger=self.logger)

                if self.use_batch_norm:
                    s10_conv3 = tf_utils.norm(s10_conv3, name='s10_norm2', _type='batch', _ops=self._ops,
                                             is_train=self.trainMode, logger=self.logger)
                s10_conv3 = tf_utils.relu(s10_conv3, name='relu_s10_conv3', logger=self.logger)

                output = tf_utils.conv2d(s10_conv3, output_dim=self.numClasses, k_h=1, k_w=1, d_h=1, d_w=1,
                                         padding=padding, initializer='He', name='output', logger=self.logger)
            else:
                output = tf_utils.conv2d(s9_conv3, output_dim=self.numClasses, k_h=1, k_w=1, d_h=1, d_w=1,
                                         padding=padding, initializer='He', name='output', logger=self.logger)

            return output