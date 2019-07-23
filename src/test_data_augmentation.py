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

def test_data_augmentation(dataPath, decodeImgShape, batchSize, isTrain, numImgs=5, margin=5, savePath='../debugImgs'):


    trainReader = Reader(tfrecordsFile=dataPath,
                         decodeImgShape=decodeImgShape,
                         batchSize=batchSize,
                         isTrain=isTrain)

    for item in ['test_random_translation_', 'test_random_flip_', 'test_random_brightness_', 'test_random_rotation']:
        imgOp, segImgOp, imgOriOp, segImgOriOp = None, None, None, None

        if item == 'test_random_translation_':
            imgOp, segImgOp, imgOriOp, segImgOriOp = trainReader.test_random_translation(numImgs=numImgs)
        elif item == 'test_random_flip_':
            imgOp, segImgOp, imgOriOp, segImgOriOp = trainReader.test_random_flip(numImgs=numImgs)
        elif item == 'test_random_brightness_':
            imgOp, segImgOp, imgOriOp, segImgOriOp = trainReader.test_random_brightness(numImgs=numImgs)
        elif item == 'test_random_rotation':
            imgOp, segImgOp, imgOriOp, segImgOriOp = trainReader.test_random_rotation(numImgs=numImgs)

        sess = tf.compat.v1.Session()

        # Threads for tfrecord
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        numTests = 20
        try:
            for i in range(numTests):
                img, segImg, imgOri, segImgOri = sess.run([imgOp, segImgOp, imgOriOp, segImgOriOp])
                img = np.squeeze(img).astype(np.uint8)
                segImg = np.squeeze(segImg).astype(np.uint8)
                imgOri = np.squeeze(imgOri).astype(np.uint8)
                segImgOri = np.squeeze(segImgOri).astype(np.uint8)

                h, w = imgOri.shape

                canvas = np.zeros((2 * h + 3 *margin, (numImgs + 1) * w + (numImgs + 3) * margin, 3), dtype=np.uint8)
                for j in range(numImgs+1):
                    if j == 0:
                        canvas[margin:margin+h, margin:margin+w, :] = np.dstack((imgOri, imgOri, imgOri))
                        canvas[2*margin+h:2*margin+2*h, margin:margin+w, :] = utils.convert_color_label(segImgOri)
                    else:
                        canvas[margin:margin+h, (j+1)*margin+j*w:(j+1)*margin+(j+1)*w, :] = \
                            np.dstack((img[j-1], img[j-1], img[j-1]))
                        canvas[2*margin+h:2*margin+2*h, (j+1)*margin+j*w:(j+1)*(w+margin), :] = \
                            utils.convert_color_label(segImg[j-1])

                cv2.imshow(winName, canvas)
                if cv2.waitKey(100) & 0xFF == 27:
                    exit('Esc clicked!')

                cv2.imwrite(os.path.join(savePath, item + str(i).zfill(2) + '.png'), canvas)

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
    batchSize_ = 1
    isTrain_ = True
    numImgs_ = 5

    test_data_augmentation(dataPath_, decodeImgShape_, batchSize_, isTrain_, numImgs_)
