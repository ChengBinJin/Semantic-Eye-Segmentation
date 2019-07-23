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


class Solver(object):
    def __init__(self, model):
        self.model = model

        self._init_session()
        self._init_variables()

    def _init_session(self):
        self.sess = tf.compat.v1.Session()

    def _init_variables(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())  # For tf.metrics.mean_iou

    def train(self):
        feed = {
            self.model.ratePh: 0.5  # rate: 1 - keep_prob
        }

        trainOp = self.model.trainOp
        totalLoss = self.model.totalLoss
        dataLoss = self.model.dataLoss
        regTerm = self.model.regTerm

        return self.sess.run([trainOp, totalLoss, dataLoss, regTerm], feed_dict=feed)

    def sample(self, iterTime, saveDir):
        feed = {
            self.model.ratePh: 0.5  # rate: 1 - keep_prob
        }

        img, predCls, segImg = self.sess.run([self.model.imgTrain, self.model.predClsTrain, self.model.segImgTrain],
                                             feed_dict=feed)

        self.saveImgs(img, predCls, segImg, iterTime, saveDir)

    def eval(self):
        feed = {
            self.model.ratePh: 0.  # rate: 1 - keep_prob
        }

        return self.sess.run(self.model.mIoU, feed_dict=feed)

    @staticmethod
    def saveImgs(img, predCls, segImg, iterTime, saveDir, margin=5):
        img = np.squeeze(img).astype(np.uint8)
        predCls = np.squeeze(predCls).astype(np.uint8)
        segImgs = np.squeeze(segImg).astype(np.uint8)

        numImgs, h, w = img.shape
        canvas = np.zeros((3 * h + 4 * margin, numImgs * w + (numImgs + 1) * margin, 3), dtype=np.uint8)

        for i in range(numImgs):
            canvas[margin:margin+h, (i+1)*margin+i*w:(i+1)*(margin+w), :] = \
                np.dstack((img[i], img[i], img[i]))
            canvas[2*margin+h:2*margin+2*h, (i+1)*margin+i*w:(i+1)*(margin+w), :] = \
                utils.convert_color_label(predCls[i])
            canvas[3*margin+2*h:3*margin+3*h, (i+1)*margin+i*w:(i+1)*(margin+w), :] = \
                utils.convert_color_label(segImgs[i])

        cv2.imwrite(os.path.join(saveDir, str(iterTime).zfill(6)+'.png'), canvas)
