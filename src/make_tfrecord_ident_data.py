# --------------------------------------------------------------------------
# Tensorflow Implementation of Eye Synthetic Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import numpy as np
import tensorflow as tf

from read_ident import get_ident_data


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('output_data', '../../Data/OpenEDS/Identification',
                       'data output directory, default: ../../Data/OpenEDS/Identification')
tf.flags.DEFINE_string('state', 'train', 'state selection from [train|validation], default: train')


def data_writer(state, outputName):
    img_paths, clses = get_ident_data(state=state)
    num_imgs = len(img_paths)

    # Create tfrecrods dir if not exists
    output_file = '{0}/{1}/{1}.tfrecords'.format(outputName, state)
    if not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Dump to tfrecords file
    writer = tf.io.TFRecordWriter(output_file)

    for idx, (img_path, cls) in enumerate(zip(img_paths, clses)):
        with tf.io.gfile.GFile(img_path, 'rb') as f:
            img_data = f.read()

        example = _convert_to_example(img_path, img_data, cls)
        writer.write(example.SerializeToString())

        if np.mod(idx, 100) == 0:
            print('Processed {}/{}...'.format(idx, num_imgs))

    print('Finished!')
    writer.close()


def _convert_to_example(img_path, img_buffer, cls):
    # Build an example proto
    img_name = os.path.basename(img_path)

    example = tf.train.Example(features=tf.train.Features(
        feature={'image/img_name': _bytes_feature(tf.compat.as_bytes(img_name)),
                 'image/cls_number': _bytes_feature(tf.compat.as_bytes(str(cls))),
                 'image/encoded_image': _bytes_feature(img_buffer)}))

    return example


def _bytes_feature(value):
    # Wrapper for inserting bytes features into example proto
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(_):
    print("Convert Identification {} data to tfrecrods...".format(FLAGS.state))
    data_writer(FLAGS.state, FLAGS.output_data)


if __name__ == '__main__':
    tf.compat.v1.app.run()
