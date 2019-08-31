# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Synthetic Eye Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------

import tensorflow as tf


class Solver(object):
    def __init__(self, model, data, is_train=True):
        self.model = model
        self.data = data
        self._init_session()
        self._init_variables()

    def _init_session(self):
        self.sess = tf.compat.v1.Session()

    def _init_variables(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self):
        def feed_run():
            fake_pair = self.sess.run(self.model.fake_pair, feed_dict={self.model.rate_tfph: 0.5})
            feed = {self.model.rate_tfph: 0.5,
                    self.model.fake_pair_tfph: self.model.img_pool_obj.query(fake_pair)}
            return feed

        self.sess.run(self.model.dis_optim, feed_dict=feed_run())
        self.sess.run(self.model.gen_optim, feed_dict=feed_run())

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, g_loss, g_adv_loss, g_cond_loss, d_loss, summary = self.sess.run(
            [self.model.gen_optim, self.model.gen_loss, self.model.gen_adv_loss, self.model.cond_loss,
             self.model.dis_loss, self.model.summary_op], feed_dict=feed_run())

        return g_loss, g_adv_loss, g_cond_loss, d_loss, summary

