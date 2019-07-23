# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------

import os
import cv2
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

import utils as utils
from dataset import Dataset
from model import UNet
from solver import Solver


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('dataset', 'OpenEDS', 'dataset name, default: OpenEDS')
tf.flags.DEFINE_string('method', 'U-Net', 'Segmentation model [U-Net, VAE], default: U-Net')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size for one iteration, default: 16')
tf.flags.DEFINE_float('resize_factor', 0.5, 'resize original input image, default: 0.5')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate for optimizer, default: 0.001')
tf.flags.DEFINE_float('weight_decay', 1e-4, 'weight decay for model to handle overfitting, default: 0.0001')
tf.flags.DEFINE_integer('iters', 100, 'number of iterations, default: 200,000')
tf.flags.DEFINE_integer('print_freq', 10, 'print frequency for loss information, default: 10')
tf.flags.DEFINE_integer('sample_freq', 20, 'sample frequence for checking qualitative evaluation, default: 100')
tf.flags.DEFINE_integer('eval_freq', 20, 'evaluation frequencey for evaluation of the batch accuracy, default: 200')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model taht you wish to continue training '
                                           '(e.g. 20190719-1409), default: None')


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize model and log folders
    if FLAGS.load_model is None:
        curTime = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        curTime = FLAGS.load_model

    modelDir, logDir, sampleDir, testDir = utils.make_folders(isTrain=FLAGS.is_train, curTime=curTime)

    # Logger
    logger = logging.getLogger(__name__)  # logger
    logger.setLevel(logging.INFO)
    utils.init_logger(logger=logger, logDir=logDir, isTrain=FLAGS.is_train, name='main')
    utils.print_main_parameters(logger, flags=FLAGS, isTrain=FLAGS.is_train)

    # Initialize dataset
    data = Dataset(name=FLAGS.dataset, isTrain=FLAGS.is_train, resizedFactor=FLAGS.resize_factor, logDir=logDir)

    # Initialize model
    model = None
    if FLAGS.method == 'U-Net':
        model = UNet(decodeImgShape=data.decodeImgShape,
                     outputShape=data.singleImgShape,
                     numClasses=data.numClasses,
                     dataPath=data(isTrain=FLAGS.is_train),
                     batchSize=FLAGS.batch_size,
                     lr=FLAGS.learning_rate,
                     weightDecay=FLAGS.weight_decay,
                     totalIters=FLAGS.iters,
                     isTrain=FLAGS.is_train,
                     logDir=logDir,
                     name='UNet')

    solver = Solver(model)

    if FLAGS.is_train is True:
        train(solver, modelDir, sampleDir)
    else:
        test()

def train(solver, modelDir, sampleDir):
    # Threads for tfrecord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=solver.sess, coord=coord)

    try:
        for iterTime in range(FLAGS.iters):
            _, totalLoss, dataLoss, regTerm = solver.train()

            if iterTime % FLAGS.print_freq == 0:
                msg = "[{0:6} / {1:6}] \tTotal loss: {2:.3f}, \tData loss: {3:.3f}, \tReg. term: {4:.3f}"
                print(msg.format(iterTime, FLAGS.iters, totalLoss, dataLoss, regTerm))

            if iterTime % FLAGS.sample_freq == 0:
                solver.sample(iterTime, sampleDir)

            if iterTime % FLAGS.eval_freq == 0:
                mIoU = solver.eval()
                print('mIoU: {}'.format(mIoU))


    except KeyboardInterrupt:
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        # when done, ask the threads to stop
        coord.request_stop()
        coord.join(threads)

def test():
    print("Hello test!")


if __name__ == '__main__':
    tf.compat.v1.app.run()
