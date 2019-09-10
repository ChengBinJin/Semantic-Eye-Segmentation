# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------

import os

import cv2
import numpy as np
import tensorflow as tf

import utils as utils
from reader import Reader

def test_advanced_multi_test(dataPath, decodeImgShape, margin=5, savePath='../debugImgs'):
    testReader = Reader(tfrecordsFile=dataPath, decodeImgShape=decodeImgShape)
    imgOp, segImgOp = testReader.test_advanced_multi_test()

    sess = tf.compat.v1.Session()

    # Threads for tfrecord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    numTests = 20
    try:
        for i in range(numTests):
            print('Processing [{:2} / {:2}]...'.format(i, numTests))

            img, segImg = sess.run([imgOp, segImgOp])
            # print('img shape: {}'.format(img.shape))
            # print('segImg shape: {}'.format(segImg.shape))

            img = np.squeeze(img).astype(np.uint8)
            segImg = np.squeeze(segImg).astype(np.uint8)

            num_imgs, h, w = img.shape
            w_num_imgs = 11
            h_num_imgs = int(num_imgs / w_num_imgs)

            img_canvas = np.zeros((h_num_imgs*h+(1+h_num_imgs)*margin, w_num_imgs*w+(w_num_imgs+1)*margin),
                                  dtype=np.uint8)
            seg_canvas = np.zeros((h_num_imgs*h+(1+h_num_imgs)*margin, w_num_imgs*w+(w_num_imgs+1)*margin, 3),
                                  dtype=np.uint8)
            for j in range(num_imgs):
                x_idx = j // w_num_imgs
                y_idx = j % w_num_imgs

                img_canvas[(x_idx+1)*margin+x_idx*h:(x_idx+1)*margin+(x_idx+1)*h,
                (y_idx+1)*margin+y_idx*w:(y_idx+1)*margin+(y_idx+1)*w] = img[j]

                seg_canvas[(x_idx+1)*margin+x_idx*h:(x_idx+1)*margin+(x_idx+1)*h,
                (y_idx+1)*margin+y_idx*w:(y_idx+1)*margin+(y_idx+1)*w] = utils.convert_color_label(segImg[j])

            cv2.imwrite(os.path.join(savePath, 'test_advanced_multi_test_img_' + str(i).zfill(2) + '.png'), img_canvas)
            cv2.imwrite(os.path.join(savePath, 'test_advanced_multi_test_seg_' + str(i).zfill(2) + '.png'), seg_canvas)
        print('[!] Finished!')

    except KeyboardInterrupt:
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)

    sess.close()


if __name__ == '__main__':
    dataPath_ = '../../Data/OpenEDS/Semantic_Segmentation_Dataset/train/train.tfrecords'
    decodeImgShape_ = (320, 400, 1)

    test_advanced_multi_test(dataPath_, decodeImgShape_)
