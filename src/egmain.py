# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Synthetic Eye Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------

import logging
import os
import tensorflow as tf
from datetime import datetime

import utils as utils
from dataset import Dataset
from pix2pix import Pix2pix


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('dataset', 'OpenEDS', 'dataset name, default: OpenEDS')
tf.flags.DEFINE_string('method', 'pix2pix', 'Generative model [pix2pix], default: pix2pix')
tf.flags.DEFINE_integer('batch_size', 1, 'batch size for one iteration, default: 1')
tf.flags.DEFINE_float('resize_factor', 0.5, 'resize original input image, default: 0.5')
tf.flags.DEFINE_float('lambda_1', 10., 'hyper-paramter for L1 loss of the conditional loss, default: 10.')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for optimizer, default: 0.0002')
tf.flags.DEFINE_integer('iters', 20, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 2, 'print frequency for loss information, default: 50')
tf.flags.DEFINE_integer('sample_freq', 5, 'sample frequence for checking qualitative evaluation, default: 500')
tf.flags.DEFINE_integer('sample_batch', 4, 'number of sampling images for check generator quality, default: 4')
tf.flags.DEFINE_integer('save_freq', 20000, 'save frequency for model, default: 20000')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20190801-220902), default: None')


def print_main_parameters(logger, flags, is_train=False):
    if is_train:
        logger.info('gpu_index: \t\t\t{}'.format(flags.gpu_index))
        logger.info('dataset: \t\t\t{}'.format(flags.dataset))
        logger.info('method: \t\t\t{}'.format(flags.method))
        logger.info('batch_size: \t\t\t{}'.format(flags.batch_size))
        logger.info('resize_factor: \t\t{}'.format(flags.resize_factor))
        logger.info('lambda_1: \t\t{}'.format(flags.lambda_1))
        logger.info('is_train: \t\t\t{}'.format(flags.is_train))
        logger.info('learning_rate: \t\t{}'.format(flags.learning_rate))
        logger.info('iters: \t\t\t{}'.format(flags.iters))
        logger.info('print_freq: \t\t\t{}'.format(flags.print_freq))
        logger.info('sample_freq: \t\t{}'.format(flags.sample_freq))
        logger.info('sample_batch: \t\t{}'.format(flags.sample_batch))
        logger.info('save_freq: \t\t\t{}'.format(flags.save_freq))
        logger.info('load_model: \t\t\t{}'.format(flags.load_model))
    else:
        print('-- gpu_index: \t\t\t{}'.format(flags.gpu_index))
        print('-- dataset: \t\t\t{}'.format(flags.dataset))
        print('-- method: \t\t\t{}'.format(flags.method))
        print('-- batch_size: \t\t\t{}'.format(flags.batch_size))
        print('-- resize_factor: \t\t{}'.format(flags.resize_factor))
        print('-- lambda_1: \t\t{}'.format(flags.lambda_1))
        print('-- is_train: \t\t\t{}'.format(flags.is_train))
        print('-- learning_rate: \t\t{}'.format(flags.learning_rate))
        print('-- iters: \t\t\t{}'.format(flags.iters))
        print('-- print_freq: \t\t\t{}'.format(flags.print_freq))
        print('-- sample_freq: \t\t{}'.format(flags.sample_freq))
        print('-- sample_batch: \t\t{}'.format(flags.sample_batch))
        print('-- save_freq: \t\t\t{}'.format(flags.save_freq))
        print('-- load_model: \t\t\t{}'.format(flags.load_model))


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize model and log folders:
    if FLAGS.load_model is None:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        cur_time = FLAGS.load_model

    model_dir, log_dir, sample_dir, _, test_dir = utils.make_folders(isTrain=FLAGS.is_train,
                                                                     curTime=cur_time,
                                                                     subfolder=os.path.join('generation', FLAGS.method))

    # Logger
    logger = logging.getLogger(__name__)  # logger
    logger.setLevel(logging.INFO)
    utils.init_logger(logger=logger, logDir=log_dir, isTrain=FLAGS.is_train, name='egmain')
    print_main_parameters(logger, flags=FLAGS, is_train=FLAGS.is_train)

    # Initialize dataset
    data = Dataset(name=FLAGS.dataset,
                   track='Generative_Dataset',
                   isTrain=FLAGS.is_train,
                   resizedFactor=FLAGS.resize_factor,
                   logDir=log_dir)

    # Initialize model
    model = Pix2pix(decode_img_shape=data.decode_img_shape,
                    output_shape=data.single_img_shape,
                    num_classes=data.num_classes,
                    data_path=data(is_train=FLAGS.is_train),
                    batch_size=FLAGS.batch_size,
                    lr=FLAGS.learning_rate,
                    total_iters=FLAGS.iters,
                    is_train=FLAGS.is_train,
                    log_dir=log_dir,
                    resize_factor=FLAGS.resize_factor,
                    lambda_1=FLAGS.lambda_1)


if __name__ == '__main__':
    tf.compat.v1.app.run()
