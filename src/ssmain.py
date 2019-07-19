# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------

import os
import logging
import tensorflow as tf
from datetime import datetime

import utils as utils
from dataset import Dataset
from model import UNet

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('dataset', 'OpenEDS', 'dataset name, default: OpenEDS')
tf.flags.DEFINE_string('method', 'U-Net', 'Segmentation model [U-Net, VAE], default: U-Net')
tf.flags.DEFINE_integer('batch_size', 2, 'batch size for one iteration, default: 16')
tf.flags.DEFINE_float('resize_factor', 0.5, 'resize original input image, default: 0.5')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate for optimizer, default: 0.001')
tf.flags.DEFINE_float('weight_decay', 1e-4, 'weight decay for model to handle overfitting, default: 0.0001')
tf.flags.DEFINE_integer('iters', 20, 'number of iterations, default: 200,000')
tf.flags.DEFINE_integer('print_freq', 10, 'print frequency for loss information, default: 10')
tf.flags.DEFINE_integer('sample_freq', 200, 'sample frequence for checking qualitative evaluation, default: 100')
tf.flags.DEFINE_integer('eval_freq', 2000, 'evaluation frequencey for evaluation of the batch accuracy, default: 200')
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
    if FLAGS.method == 'U-Net':
        model = UNet(imageShape=data.imgShape,
                     outputShape=data.singleImgShape,
                     resizeFactor=FLAGS.resize_factor,
                     dataPath=data(isTrain=FLAGS.is_train),
                     batchSize=FLAGS.batch_size,
                     lr=FLAGS.learning_rate,
                     weight_decay=FLAGS.weight_decay,
                     total_iters=FLAGS.iters,
                     isTrain=FLAGS.is_train,
                     logDir=logDir,
                     name='UNet')


if __name__ == '__main__':
    tf.compat.v1.app.run()
