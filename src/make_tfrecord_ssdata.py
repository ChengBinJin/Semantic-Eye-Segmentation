# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import numpy as np
import tensorflow as tf

import utils as utils


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('input_data', '../../Data/OpenEDS/Semantic_Segmentation_Dataset',
                       'data input directory, default: ../../Data/OpenEDS/Semantic_Segmentation_Dataset')
tf.flags.DEFINE_string('output_data', '../../Data/OpenEDS/Semantic_Segmentation_Dataset',
                       'data output directory, default: ../../Data/OpenEDS/Semantic_Segmentation_Dataset')
tf.flags.DEFINE_string('stage', 'train', 'stage selection from [train|validation|test|overfitting], default: train')


def data_writer(inputDir, stage, outputName):
    dataPath = os.path.join(inputDir, '{}'.format(stage), 'paired')
    imgPaths = utils.all_files_under(folder=dataPath, subfolder='')
    numImgs = len(imgPaths)

    # Create tfrecrods dir if not exists
    output_file = '{0}/{1}/{1}.tfrecords'.format(outputName, stage)
    if not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Dump to tfrecords file
    writer = tf.io.TFRecordWriter(output_file)

    for idx, img_path in enumerate(imgPaths):
        with tf.io.gfile.GFile(img_path, 'rb') as f:
            img_data = f.read()

        example = _convert_to_example(img_path, img_data)
        writer.write(example.SerializeToString())

        if np.mod(idx, 100) == 0:
            print('Processed {}/{}...'.format(idx, numImgs))

    print('Finished!')
    writer.close()


def _convert_to_example(imgPath, imgBuffer):
    # Build an example proto
    imgName = os.path.basename(imgPath)
    userId = imgName.replace('.png', '').split('_')[1]

    example = tf.train.Example(features=tf.train.Features(
        feature={'image/file_name': _bytes_feature(tf.compat.as_bytes(imgName)),
                 'image/user_id': _bytes_feature(tf.compat.as_bytes(userId)),
                 'image/encoded_image': _bytes_feature(imgBuffer)}))

    return example


def _bytes_feature(value):
    # Wrapper for inserting bytes features into example proto
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(_):
    print("Convert {} - {} data to tfrecrods...".format(FLAGS.input_data, FLAGS.stage))
    data_writer(FLAGS.input_data, FLAGS.stage, FLAGS.output_data)


if __name__ == '__main__':
    tf.compat.v1.app.run()
