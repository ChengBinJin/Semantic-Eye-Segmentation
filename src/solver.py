# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import utils as utils


class Solver(object):
    def __init__(self, model, data, batchSize):
        self.model = model
        self.data = data
        self.batchSize = batchSize

        self._init_session()
        self._init_variables()

    def _init_session(self):
        self.sess = tf.compat.v1.Session()

    def _init_variables(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self):
        feed = {
            self.model.ratePh: 0.5  # rate: 1 - keep_prob
        }

        trainOp = self.model.trainOp
        totalLoss = self.model.totalLoss
        dataLoss = self.model.dataLoss
        regTerm = self.model.regTerm
        summary_op = self.model.summary_op

        return self.sess.run([trainOp, totalLoss, dataLoss, regTerm, summary_op], feed_dict=feed)

    def sample(self, iterTime, saveDir):
        feed = {
            self.model.ratePh: 0.5  # rate: 1 - keep_prob
        }

        img, predCls, segImg = self.sess.run([self.model.imgTrain, self.model.predClsTrain, self.model.segImgTrain],
                                             feed_dict=feed)

        self.saveImgs(img, predCls, segImg, iterTime, saveDir)

    def eval(self, tb_writer, iter_time):
        # Calculate number of iterations for one validaiton-epoch
        numIters = int(np.ceil(self.data.numValImgs / self.batchSize))

        feed = {
            self.model.ratePh: 0.  # rate: 1 - keep_prob
        }

        # Initialize/reset the running variables
        self.sess.run(self.model.running_vars_initializer)

        for iterTime in range(numIters):
            # Update the running variables on new batch of samples
            self.sess.run([self.model.mIoU_metric_update, self.model.accuracy_metric_update], feed_dict=feed)

            if iterTime % 100 == 0:
                msg  = "\r - Evaluating progress: {:.2f}%".format((iterTime/numIters)*100.)

                # Print it.
                sys.stdout.write(msg)
                sys.stdout.flush()

        # Calculate the mIoU
        mIoU, accuracy, metric_summary_op = self.sess.run([self.model.mIoU_metric,
                                                           self.model.accuracy_metric,
                                                           self.model.metric_summary_op])

        # Write to tensorboard
        tb_writer.add_summary(metric_summary_op, iter_time)
        tb_writer.flush()

        return mIoU * 100., accuracy * 100.

    def set_best_mIoU(self, best_mIoU):
        self.sess.run(self.model.assign_best_mIoU, feed_dict={self.model.best_mIoU_ph: best_mIoU})

    def get_best_mIoU(self):
        return self.sess.run(self.model.best_mIoU)

    def set_best_acc(self, best_acc):
        self.sess.run(self.model.assign_best_acc, feed_dict={self.model.best_acc_ph: best_acc})

    def get_best_acc(self):
        return self.sess.run(self.model.best_acc)

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
