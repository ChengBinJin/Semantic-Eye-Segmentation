# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------

import os
import logging
import tensorflow as tf
from datetime import datetime

import utils as utils
from dataset import Dataset
from model import UNet
from solver import Solver


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index if you have multiple gpus, default: 0')
tf.flags.DEFINE_string('dataset', 'OpenEDS', 'dataset name, default: OpenEDS')
tf.flags.DEFINE_string('method', 'U-Net', 'Segmentation model [U-Net, VAE], default: U-Net')
tf.flags.DEFINE_integer('batch_size', 16, 'batch size for one iteration, default: 16')
tf.flags.DEFINE_float('resize_factor', 0.5, 'resize original input image, default: 0.5')
tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate for optimizer, default: 0.001')
tf.flags.DEFINE_float('weight_decay', 1e-4, 'weight decay for model to handle overfitting, default: 0.0001')
tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200,000')
tf.flags.DEFINE_integer('print_freq', 10, 'print frequency for loss information, default: 10')
tf.flags.DEFINE_integer('sample_freq', 100, 'sample frequence for checking qualitative evaluation, default: 100')
tf.flags.DEFINE_integer('eval_freq', 2000, 'evaluation frequencey for evaluation of the batch accuracy, default: 2000')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model taht you wish to continue training '
                                           '(e.g. 20190719-1409), default: None')


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_index

    # Initialize model and log folders
    if FLAGS.load_model is None:
        curTime = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        curTime = FLAGS.load_model

    modelDir, logDir, sampleDir, valDir, testDir = utils.make_folders(isTrain=FLAGS.is_train, curTime=curTime)

    # Logger
    logger = logging.getLogger(__name__)  # logger
    logger.setLevel(logging.INFO)
    utils.init_logger(logger=logger, logDir=logDir, isTrain=FLAGS.is_train, name='main')
    utils.print_main_parameters(logger, flags=FLAGS, isTrain=FLAGS.is_train)

    # Initialize dataset
    data = Dataset(name=FLAGS.dataset, isTrain=FLAGS.is_train, resizedFactor=FLAGS.resize_factor, logDir=logDir)

    # Initialize model
    model = None
    if FLAGS.method == 'U-Net':
        model = UNet(decodeImgShape=data.decodeImgShape,
                     outputShape=data.singleImgShape,
                     numClasses=data.numClasses,
                     dataPath=data(isTrain=FLAGS.is_train),
                     batchSize=FLAGS.batch_size if FLAGS.is_train else 1,
                     lr=FLAGS.learning_rate,
                     weightDecay=FLAGS.weight_decay,
                     totalIters=FLAGS.iters,
                     isTrain=FLAGS.is_train,
                     logDir=logDir,
                     name='UNet')

    # Initialize solver
    solver = Solver(model=model,
                    data=data,
                    batchSize=FLAGS.batch_size if FLAGS.is_train else 1)

    # Initialize saver
    saver = tf.compat.v1.train.Saver(max_to_keep=1)

    if FLAGS.is_train is True:
        train(solver, saver, logger, modelDir, logDir, sampleDir)
    else:
        test(solver, saver, modelDir, valDir, testDir, data)

def train(solver, saver, logger, modelDir, logDir, sampleDir):
    best_mIoU, best_acc, best_precision, best_recall = 0., 0., 0., 0.
    iterTime = 0

    if FLAGS.load_model is not None:
        flag, iterTime, best_mIoU, best_acc, best_precision, best_recall = load_model(saver=saver,
                                                                                      solver=solver,
                                                                                      logger=logger,
                                                                                      model_dir=modelDir,
                                                                                      is_train=True)
        if flag is True:
            logger.info(' [!] Load Sucess! Iter: {}'.format(iterTime))
            logger.info('Best mIoU: {:.3f}'.format(best_mIoU))
            logger.info('Best Acc.: {:.3f}'.format(best_acc))
            logger.info('Best Precison: {:.3f}'.format(best_precision))
            logger.info('Best Recall: {:.3f}'.format(best_recall))
        else:
            exit(" [!] Failed to restore model {}".format(FLAGS.load_model))

    # Tensorboard writer
    tb_writer = tf.compat.v1.summary.FileWriter(logdir=logDir, graph=solver.sess.graph_def)

    # Threads for tfrecord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=solver.sess, coord=coord)

    try:
        while iterTime < FLAGS.iters:
            _, total_loss, data_loss, reg_term, summary = solver.train()

            # Write to tensorboard
            tb_writer.add_summary(summary, iterTime)
            tb_writer.flush()

            # Print loss information
            if (iterTime % FLAGS.print_freq == 0) or (iterTime + 1 == FLAGS.iters):
                msg = "[{0:6} / {1:6}] Total loss: {2:.3f}, Data loss: {3:.3f}, Reg. term: {4:.3f}"
                print(msg.format(iterTime, FLAGS.iters, total_loss, data_loss, reg_term))

            # Sampling predictive results
            if (iterTime % FLAGS.sample_freq == 0) or (iterTime + 1 == FLAGS.iters):
                solver.sample(iterTime, sampleDir)

            # Evaluat models using validation dataset
            if (iterTime % FLAGS.eval_freq == 0) or (iterTime + 1 == FLAGS.iters):
                mIoU, acc, precision, recall = solver.eval(tb_writer=tb_writer, iter_time=iterTime)

                if best_acc < acc:
                    best_acc = acc
                    solver.set_best_acc(best_acc)

                if best_precision < precision:
                    best_precision = precision
                    solver.set_best_precision(best_precision)

                if best_recall < recall:
                    best_recall = recall
                    solver.set_best_recall(best_recall)

                if best_mIoU < mIoU:
                    best_mIoU = mIoU
                    solver.set_best_mIoU(best_mIoU)
                    save_model(saver, solver, logger, modelDir, iterTime, best_mIoU)

                print("\n")
                print("*"*70)
                print('mIoU: {:.3f} \t\t- Best mIoU: {:.3f}'.format(mIoU, best_mIoU))
                print('Acc.: {:.3f} \t\t- Best Acc.: {:.3f}'.format(acc, best_acc))
                print("Precision: {:.3f} \t- Best Precision: {:.3f}".format(precision, best_precision))
                print("Recall: {:.3f} \t\t- Best Recall: {:.3f}".format(recall, best_recall))
                print("*"*70)

            iterTime += 1

    except KeyboardInterrupt:
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    except tf.errors.OutOfRangeError:
        coord.request_stop()
    finally:
        # when done, ask the threads to stop
        coord.request_stop()
        coord.join(threads)

def test(solver, saver, modelDir, valDir, testDir, data):
    # Load checkpoint
    flag, iter_time, best_mIoU, best_acc, best_precision, best_recall = load_model(saver=saver,
                                                                                   solver=solver,
                                                                                   logger=None,
                                                                                   model_dir=modelDir,
                                                                                   is_train=False)
    if flag is True:
        print(' [!] Load Success! Iter: {}, Best mIoU: {:.3f}'.format(iter_time, best_mIoU))
    else:
        exit(' [!] Load Failed! Can not find model {}'.format(FLAGS.load_model))

    # Threads for tfrecord
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=solver.sess, coord=coord)

    try:
        # mIoU, acc, precision, recall = solver.test_val(save_dir=valDir)
        #
        # print("\n")
        # print("*" * 70)
        # print('mIoU: {:.3f} \t\t- Best mIoU: {:.3f}'.format(mIoU, best_mIoU))
        # print('Acc.: {:.3f} \t\t- Best Acc.: {:.3f}'.format(acc, best_acc))
        # print("Precision: {:.3f} \t- Best Precision: {:.3f}".format(precision, best_precision))
        # print("Recall: {:.3f} \t\t- Best Recall: {:.3f}".format(recall, best_recall))
        # print("*" * 70)

        solver.test_test(save_dir=testDir)

    except KeyboardInterrupt:
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    except tf.errors.OutOfRangeError:
        coord.request_stop()
    finally:
        # when done, ask the threads to stop
        coord.request_stop()
        coord.join(threads)

def save_model(saver, solver, logger, model_dir, iter_time, best_mIoU):
    saver.save(solver.sess, os.path.join(model_dir, 'model'), global_step=iter_time)
    logger.info('\n [*] Model saved! Iter: {}, Best mIoU: {:.3f}'.format(iter_time, best_mIoU))


def load_model(saver, solver, logger, model_dir, is_train=False):
    if is_train:
        logger.info(' [*] Reading checkpoint...')
    else:
        print(' [*] Reading checkpoint...')

    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(solver.sess, os.path.join(model_dir, ckpt_name))

        meta_graph_path = ckpt.model_checkpoint_path + '.meta'
        iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

        if is_train:
            logger.info(' [!] Load Iter: {}'.format(iter_time))
        else:
            print(' [!] Load Iter: {}'.format(iter_time))

        # Get metrics from the model checkpoints
        best_mIoU = solver.get_best_mIoU()
        best_acc = solver.get_best_acc()
        best_precision = solver.get_best_precision()
        best_recall = solver.get_best_recall()

        return True, iter_time + 1, best_mIoU, best_acc, best_precision, best_recall
    else:
        return False, None, None, None, None, None


if __name__ == '__main__':
    tf.compat.v1.app.run()
