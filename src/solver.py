# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import sys
import numpy as np
import tensorflow as tf
import utils as utils


class Solver(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data

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

    def eval(self, tb_writer=None, iter_time=None, save_dir=None, is_test=False):
        run_ops = [self.model.mIoU_metric_update,
                   self.model.accuracy_metric_update,
                   self.model.precision_metric_update,
                   self.model.recall_metric_update,
                   self.model.per_class_accuracy_metric_update,
                   self.model.imgVal,
                   self.model.predClsVal,
                   self.model.segImgVal,
                   self.model.img_name_val,
                   self.model.user_id_val]

        feed = {
            self.model.ratePh: 0.  # rate: 1 - keep_prob
        }

        # Initialize/reset the running variables
        self.sess.run(self.model.running_vars_initializer)

        per_cla_acc_mat = None
        for iterTime in range(self.data.numValImgs):
            # Update the running variables on new batch of samples
            _, _, _, _, per_cla_acc_mat, img, predCls, segImg, img_name, user_id = self.sess.run(
                run_ops, feed_dict=feed)

            if iterTime % 100 == 0:
                msg  = "\r - Evaluating progress: {:.2f}%".format((iterTime/self.data.numValImgs)*100.)

                # Print it.
                sys.stdout.write(msg)
                sys.stdout.flush()

            if is_test:
                # Save images
                utils.save_imgs(img_stores=[img, predCls, segImg],
                                    saveDir=save_dir,
                                    img_name=img_name.astype('U26'),
                                    is_vertical=False)

        # Calculate the mIoU
        mIoU, accuracy, precision, recall, f1_score,  metric_summary_op = self.sess.run([
            self.model.mIoU_metric,
            self.model.accuracy_metric,
            self.model.precision_metric,
            self.model.recall_metric,
            self.model.f1_score_metric,
            self.model.metric_summary_op])

        if not is_test:
            # Write to tensorboard
            tb_writer.add_summary(metric_summary_op, iter_time)
            tb_writer.flush()

        mIoU *= 100.
        accuracy *= 100.
        precision *= 100.
        recall *= 100.
        f1_score *= 100.
        per_cla_acc_mat *= 100.

        return mIoU, accuracy, per_cla_acc_mat, precision, recall, f1_score
    
    def test_test(self, save_dir):
        print('Number of iterations: {}'.format(self.data.numTestImgs))

        run_ops = [self.model.imgTest,
                   self.model.predClsTest,
                   self.model.img_name_test,
                   self.model.user_id_test]

        feed = {
            self.model.ratePh: 0.  # rate: 1 - keep_prob
        }

        for iterTime in range(self.data.numTestImgs):
            img, predCls, img_name, user_id = self.sess.run(run_ops, feed_dict=feed)

            # Save images
            utils.save_imgs(img_stores=[img, predCls],
                            saveDir=save_dir,
                            img_name=img_name.astype('U26'),
                            is_vertical=False)

            # Write as npy format
            utils.save_npy(data=predCls,
                           save_dir=save_dir,
                           file_name=img_name.astype('U26'))

            if iterTime % 100 == 0:
                print("- Evaluating progress: {:.2f}%".format((iterTime/self.data.numTestImgs)*100.))

    def sample(self, iterTime, saveDir, num_imgs=8):
        feed = {
            self.model.ratePh: 0.5  # rate: 1 - keep_prob
        }

        img, predCls, segImg = self.sess.run([self.model.imgTrain, self.model.predClsTrain, self.model.segImgTrain],
                                             feed_dict=feed)

        # if batch_size is bigger than num_imgs, we just show num_imgs
        num_imgs = np.minimum(num_imgs, img.shape[0])

        # Save imgs
        utils.save_imgs(img_stores=[img[:num_imgs], predCls[:num_imgs], segImg[:num_imgs]],
                        iterTime=iterTime,
                        saveDir=saveDir,
                        is_vertical=True)

    def set_best_mIoU(self, best_mIoU):
        self.sess.run(self.model.assign_best_mIoU, feed_dict={self.model.best_mIoU_ph: best_mIoU})

    def get_best_mIoU(self):
        return self.sess.run(self.model.best_mIoU)

    def set_best_acc(self, best_acc):
        self.sess.run(self.model.assign_best_acc, feed_dict={self.model.best_acc_ph: best_acc})

    def get_best_acc(self):
        return self.sess.run(self.model.best_acc)

    def set_best_precision(self, best_precision):
        self.sess.run(self.model.assign_best_precision, feed_dict={self.model.best_precision_ph: best_precision})

    def get_best_precision(self):
        return self.sess.run(self.model.best_precision)

    def set_best_recall(self, best_recall):
        self.sess.run(self.model.assign_best_recall, feed_dict={self.model.best_recall_ph: best_recall})

    def get_best_recall(self):
        return self.sess.run(self.model.best_recall)

    def set_best_f1_score(self, best_f1_score):
        self.sess.run(self.model.assign_best_f1_score, feed_dict={self.model.best_f1_score_ph: best_f1_score})

    def get_best_f1_score(self):
        return self.sess.run(self.model.best_f1_score)
