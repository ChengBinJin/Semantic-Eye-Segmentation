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

winName = "Show"
cv2.namedWindow(winName)
cv2.moveWindow(winName, 10, 10)

def test_multi_test(dataPath, decodeImgShape, margin=5, savePath='../debugImgs'):


    testReader = Reader(tfrecordsFile=dataPath,
                         decodeImgShape=decodeImgShape)
    imgOp, segImgOp = testReader.test_multi_test()

    sess = tf.compat.v1.Session()

    # Threads for tfrecord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    numTests = 20
    try:
        for i in range(numTests):
            img, segImg = sess.run([imgOp, segImgOp])
            img = np.squeeze(img).astype(np.uint8)
            segImg = np.squeeze(segImg).astype(np.uint8)

            num_imgs, h, w = img.shape


            canvas = np.zeros((2*h+3 *margin, num_imgs*w+(num_imgs+2)*margin, 3), dtype=np.uint8)
            for j in range(num_imgs):
                canvas[margin:margin+h, (j+1)*margin+j*w:(j+1)*margin+(j+1)*w, :] = \
                    np.dstack((img[j], img[j], img[j]))
                canvas[2*margin+h:2*margin+2*h, (j+1)*margin+j*w:(j+1)*(w+margin), :] = \
                    utils.convert_color_label(segImg[j])

            cv2.imshow(winName, canvas)
            if cv2.waitKey(100) & 0xFF == 27:
                exit('Esc clicked!')

            cv2.imwrite(os.path.join(savePath, 'test_multi_test_' + str(i).zfill(2) + '.png'), canvas)

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

    test_multi_test(dataPath_, decodeImgShape_)
