# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import os
import cv2
import sys
import time
import numpy as np
import tensorflow as tf
import utils as utils


class Solver(object):
    def __init__(self, model, data, is_train=False, multi_test=False):
        self.model = model
        self.data = data
        self.is_train = is_train
        self.multi_test = False if self.is_train else multi_test
        self._init_session()
        self._init_variables()

    def _init_session(self):
        self.sess = tf.compat.v1.Session()

    def _init_variables(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self):
        feed = {
            self.model.ratePh: 0.5,  # rate: 1 - keep_prob
            self.model.trainMode: True
        }

        train_op = self.model.trainOp
        total_loss_op = self.model.totalLoss
        data_loss_op = self.model.dataLoss
        reg_term_op = self.model.regTerm
        dice_loss_op = self.model.dice_loss
        summary_op = self.model.summary_op

        _, total_loss, data_loss, reg_term, dice_loss, summary = self.sess.run(
            [train_op, total_loss_op, data_loss_op, reg_term_op, dice_loss_op, summary_op], feed_dict=feed)

        return total_loss, data_loss, reg_term, dice_loss, summary

    def eval(self, tb_writer=None, iter_time=None, save_dir=None, is_debug=False):
        if self.multi_test:
            run_ops = [self.model.mIoU_metric_update,
                       self.model.accuracy_metric_update,
                       self.model.precision_metric_update,
                       self.model.recall_metric_update,
                       self.model.per_class_accuracy_metric_update,
                       self.model.imgVal,
                       self.model.predClsVal,
                       self.model.segImgVal,
                       self.model.img_name_val,
                       self.model.user_id_val,
                       self.model.imgVal_s1,
                       self.model.imgVal_s2,
                       self.model.predVal_s1,
                       self.model.predVal_s2,
                       self.model.segImgVal_s1,
                       self.model.segImgVal_s2]
        else:
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
            self.model.ratePh: 0.,  # rate: 1 - keep_prob
            self.model.trainMode: False
        }

        # Initialize/reset the running variables
        self.sess.run(self.model.running_vars_initializer)

        per_cla_acc_mat = None
        for iterTime in range(self.data.numValImgs):
            img_s1, img_s2, pred_s1, pred_s2, segImg_s1, segImg_s2 = None, None, None, None, None, None

            if self.multi_test:
                _, _, _, _, per_cla_acc_mat, img, predCls, segImg, img_name, user_id, \
                img_s1, img_s2, pred_s1, pred_s2, segImg_s1, segImg_s2 = self.sess.run(run_ops, feed_dict=feed)
            else:
                _, _, _, _, per_cla_acc_mat, img, predCls, segImg, img_name, user_id = \
                    self.sess.run(run_ops, feed_dict=feed)

            if iterTime % 100 == 0:
                msg  = "\r - Evaluating progress: {:.2f}%".format((iterTime/self.data.numValImgs)*100.)

                # Print it.
                sys.stdout.write(msg)
                sys.stdout.flush()

            ############################################################################################################
            if not self.is_train:
                # Save images
                utils.save_imgs(img_stores=[img, predCls, segImg],
                                saveDir=save_dir,
                                img_name=img_name.astype('U26'),
                                is_vertical=False)

            if not self.is_train and is_debug and self.multi_test:
                # # Step 1: save rotated images
                predCls_s1 = np.argmax(pred_s1, axis=-1)  # predict class using argmax function
                utils.save_imgs(img_stores=[img_s1, predCls_s1, segImg_s1],
                                saveDir=os.path.join(save_dir, 'debug'),
                                name_append='step1_',
                                img_name=img_name.astype('U26'),
                                is_vertical=True)

                # Step 2: save inverse-roated images
                predCls_s2 = np.argmax(pred_s2, axis=-1)  # predict class using argmax function
                utils.save_imgs(img_stores=[img_s2, predCls_s2, segImg_s2],
                                saveDir=os.path.join(save_dir, 'debug'),
                                name_append='step2_',
                                img_name=img_name.astype('U26'),
                                is_vertical=True)

                # Step 3: Save comparison image that includes img, single_pred, multi-test_pred, gt
                utils.save_imgs(img_stores=[img, np.expand_dims(predCls_s1[5], axis=0), predCls, segImg],
                                saveDir=os.path.join(save_dir, 'debug'),
                                name_append='step3_',
                                img_name=img_name.astype('U26'),
                                is_vertical=False)

            ############################################################################################################

        # Calculate the mIoU
        mIoU, accuracy, precision, recall, f1_score,  metric_summary_op = self.sess.run([
            self.model.mIoU_metric,
            self.model.accuracy_metric,
            self.model.precision_metric,
            self.model.recall_metric,
            self.model.f1_score_metric,
            self.model.metric_summary_op])

        if self.is_train:
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

    def test_test(self, save_dir, is_debug=False, max_degree=10, interval=1, resize_factor=0.25):
        s2_imgs, s3_imgs, s4_imgs = None, None, None
        batch = len(range(-max_degree, max_degree+1, interval))

        for iter_time in range(self.data.numTestImgs):
            print('[*] Multi-test processing {} / {}...'.format(iter_time + 1, self.data.numTestImgs))
            s1_imgs, img_name, user_id = self.sess.run([self.model.imgTests,
                                                     self.model.img_name_test,
                                                     self.model.user_id_test])
            num_imgs, h, w, c = s1_imgs.shape

            print('[*] Stage 1: Forwar the network {} times... '.format(int(num_imgs / batch)))
            s1_preds = list()
            for i in range(0, num_imgs, batch):
                pred_imgs = self.sess.run(self.model.predTest, feed_dict={self.model.ratePh: 0.,  # keep_ratio = 1 - rate
                                                                          self.model.trainMode: False,
                                                                          self.model.inputImgPh: s1_imgs[i:i+batch]})
                s1_preds.extend(pred_imgs)
            s1_preds = np.asarray(s1_preds)

            if is_debug:
                # Save stage 1 results: the rotated, flipped, and cropped images
                utils.save_imgs_indiv(s1_imgs, w_num_imgs=batch, save_dir=os.path.join(save_dir, 'debug'),
                                      img_name=img_name.astype('U26'), is_label=False, name_append='s1_img_',
                                      factor=resize_factor)
                utils.save_imgs_indiv(s1_preds, w_num_imgs=batch, save_dir=os.path.join(save_dir, 'debug'),
                                      img_name=img_name.astype('U26'), is_label=True, name_append='s1_seg_',
                                      factor=resize_factor)

            # Save stage 2 results: the flipped and cropped images
            print('[*] Stage 2: Processing inverse-rotation...')
            s2_preds = utils.inverse_rotate(s1_preds, batch=batch, interval=interval, is_label=True)
            if is_debug:
                s2_imgs = utils.inverse_rotate(s1_imgs, batch=batch, interval=interval, is_label=False)
                utils.save_imgs_indiv(s2_imgs, w_num_imgs=batch, save_dir=os.path.join(save_dir, 'debug'),
                                      img_name=img_name.astype('U26'), is_label=False, name_append='s2_img_',
                                      factor=resize_factor)
                utils.save_imgs_indiv(s2_preds, w_num_imgs=batch, save_dir=os.path.join(save_dir, 'debug'),
                                      img_name=img_name.astype('U26'), is_label=True, name_append='s2_seg_',
                                      factor=resize_factor)


            # Save stage 3 results: the cropped images
            print('[*] Stage 3: Processing inverse-flipping...')
            s3_preds = utils.inverse_flip(s2_preds)
            if is_debug:
                s3_imgs = utils.inverse_flip(s2_imgs)
                utils.save_imgs_indiv(s3_imgs, w_num_imgs=batch, save_dir=os.path.join(save_dir, 'debug'),
                                      img_name=img_name.astype('U26'), is_label=False, name_append='s3_img_',
                                      factor=resize_factor)
                utils.save_imgs_indiv(s3_preds, w_num_imgs=batch, save_dir=os.path.join(save_dir, 'debug'),
                                      img_name=img_name.astype('U26'), is_label=True, name_append='s3_seg_',
                                      factor=resize_factor)

            # Save stage 4 results
            print('[*] Stage 4: Processing inverse-cropping...')
            s4_preds = utils.inverse_cropping(s3_preds, batch=batch, is_label=True)
            if is_debug:
                s4_imgs = utils.inverse_cropping(s3_imgs, batch=batch, is_label=False)
                utils.save_imgs_indiv(s4_imgs, w_num_imgs=batch, save_dir=os.path.join(save_dir, 'debug'),
                                      img_name=img_name.astype('U26'), is_label=False, name_append='s4_img_',
                                      factor=resize_factor)
                utils.save_imgs_indiv(s4_preds, w_num_imgs=batch, save_dir=os.path.join(save_dir, 'debug'),
                                      img_name=img_name.astype('U26'), is_label=True, name_append='s4_seg_',
                                      factor=resize_factor)

            # Save stage 5 results
            print('[*] Stage 5: Calculaing the final prediction...')
            predCls = np.argmax(np.sum(s4_preds, axis=0, keepdims=True), axis=3)
            utils.save_imgs(img_stores=[np.expand_dims(s1_imgs[int(batch*0.5)], axis=0),
                                        np.expand_dims(np.argmax(s4_preds[int(batch*0.5)], axis=2), axis=0),
                                        predCls],
                            saveDir=os.path.join(save_dir, 'debug'),
                            img_name=img_name.astype('U26'),
                            is_vertical=False)

            # Save images
            print('[*] Saving the input and prediction image...')
            utils.save_imgs(img_stores=[np.expand_dims(s1_imgs[int(batch*0.5)], axis=0), predCls],
                            saveDir=save_dir,
                            img_name=img_name.astype('U26'),
                            is_vertical=False)

            # Write as npy format
            print('[*] Saving .npy files for submission...')
            utils.save_npy(data=predCls,
                           save_dir=save_dir,
                           file_name=img_name.astype('U26'))

    
    # def test_test(self, save_dir, is_debug=True):
    #     print('Number of iterations: {}'.format(self.data.numTestImgs))
    #
    #     if self.multi_test:
    #         run_ops = [self.model.imgTest,
    #                    self.model.predTest_s1,
    #                    self.model.predClsTest,
    #                    self.model.img_name_test,
    #                    self.model.user_id_test]
    #     else:
    #         run_ops = [self.model.imgTest,
    #                    self.model.predClsTest,
    #                    self.model.img_name_test,
    #                    self.model.user_id_test]
    #
    #     feed = {
    #         self.model.ratePh: 0.,  # rate: 1 - keep_prob
    #         self.model.trainMode: False
    #     }
    #
    #     # Time check
    #     total_time = 0.
    #
    #     pred_s1 = None
    #     for iterTime in range(self.data.numTestImgs):
    #         tic = time.time()  # tic
    #
    #         if self.multi_test:
    #             img, pred_s1, predCls, img_name, user_id = self.sess.run(run_ops, feed_dict=feed)
    #         else:
    #             img, predCls, img_name, user_id = self.sess.run(run_ops, feed_dict=feed)
    #
    #         toc = time.time()  # toc
    #         total_time += toc - tic
    #
    #         # Debug for multi-test
    #         if self.multi_test and is_debug:
    #             predCls_s1 = np.argmax(pred_s1, axis=-1)  # predict class using argmax function
    #
    #             # Save images
    #             utils.save_imgs(img_stores=[img, np.expand_dims(predCls_s1[5], axis=0), predCls],
    #                             saveDir=os.path.join(save_dir, 'debug'),
    #                             img_name=img_name.astype('U26'),
    #                             is_vertical=False)
    #
    #         # Save images
    #         utils.save_imgs(img_stores=[img, predCls],
    #                         saveDir=save_dir,
    #                         img_name=img_name.astype('U26'),
    #                         is_vertical=False)
    #
    #         # Write as npy format
    #         utils.save_npy(data=predCls,
    #                        save_dir=save_dir,
    #                        file_name=img_name.astype('U26'))
    #
    #         if iterTime % 100 == 0:
    #             print("- Evaluating progress: {:.2f}%".format((iterTime/self.data.numTestImgs)*100.))
    #
    #     msg = "Average processing time: {:.2f} msec. for one image"
    #     print(msg.format(total_time / self.data.numTestImgs * 1000.))

    def sample(self, iterTime, saveDir, num_imgs=4):
        feed = {
            self.model.ratePh: 0.5,  # rate: 1 - keep_prob
            self.model.trainMode: True
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
