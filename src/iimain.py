# --------------------------------------------------------------------------
# Tensorflow Implementation of Synthetic Eye Generation Challenge
# Iris Identification
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------

import os
import logging
from datetime import datetime
import tensorflow as tf
import utils as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_integer('mode', 0, 'mode 0: the gray-scale image only, '
                                   'mode 1: the grapy-scal iamge + cropped iris image using segmentation map,'
                                   'mode 2: the cropped iris image only,'
                                   'default: 0')
tf.flags.DEFINE_string('method', 'recognition', '[recognition, verification], default: recognition')
tf.flags.DEFINE_integer('batch_size', 2, 'batch size for one iteration, default: 128')
tf.flags.DEFINE_float('resize_factor', 0.5, 'resize the original input image, default: 0.5')

tf.flags.DEFINE_string('dataset', 'OpenEDS', 'dataset name, default: OpenEDS')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('leanring_rate', 1e-3, 'initial learning rate for optimizer, default: 0.001')
tf.flags.DEFINE_float('weight_decay', 1e-4, 'weight decay for model to handle overfitting, default: 0.0001')
tf.flags.DEFINE_integer('epoch', 20, 'number of epochs, default: 20')
tf.flags.DEFINE_integer('print_freq', 5, 'print frequence for loss information, default: 50')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20190806-234308), default: None')

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize model and log folders
    if FLAGS.load_model is None:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        cur_time = FLAGS.load_model

    model_dir, log_dir = utils.make_folders_simple(cur_time=cur_time,
                                                   subfolder=os.path.join('identification', FLAGS.method))

    print("Hello iimain.py!")

if __name__ == '__main__':
    tf.compat.v1.app.run()