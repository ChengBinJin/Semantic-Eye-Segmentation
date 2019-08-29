# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import logging
import utils as utils

class OpenEDS_Generation(object):
    def __init__(self, name='OpenEDS', track='Generation', is_train=True, resized_factor=0.5, log_dir=None):
        self.name = name
        # self.track = track
        self.track = 'Semantic_Segmentation_Dataset'

        self.num_train_imgs = 8916
        self.num_val_imgs = 2403
        self.num_test_imgs = 1471

        self.num_train_persons = 95
        self.num_val_persons = 28
        self.num_test_persons = 29
        self.num_classes = 4

        self.decode_img_shape = (int(640 * resized_factor), int(400 * 2 * resized_factor), 1)
        self.single_img_shape = (int(640 * resized_factor), int(400 * resized_factor), 1)

        # TFrecord path
        self.train_path = '../../Data/OpenEDS/{}/train/train.tfrecords'.format(self.track)
        self.val_path = '../../Data/OpenEDS/{}/validation/validation.tfrecords'.format(self.track)
        self.test_path = '../../Data/OpenEDS/{}/test/test.tfrecords'.format(self.track)
        self.overfitting_path = '../../Data/OpenEDS/{}/overfitting/overfitting.tfrecords'.format(self.track)

        if is_train:
            self.logger = logging.getLogger(__name__)   # logger
            self.logger.setLevel(logging.INFO)
            utils.init_logger(logger=self.logger, logDir=log_dir, isTrain=is_train, name='dataset')

            self.logger.info('Dataset name: \t\t{}'.format(self.name))
            self.logger.info('Dataset track: \t\t{}'.format(self.track))
            self.logger.info('Num. of training imgs: \t{}'.format(self.num_train_imgs))
            self.logger.info('Num. of validation imgs: \t{}'.format(self.num_val_imgs))
            self.logger.info('Num. of test imgs: \t\t{}'.format(self.num_test_imgs))
            self.logger.info('Num. of training persons: \t{}'.format(self.num_train_persons))
            self.logger.info('Num. of validation persons: \t{}'.format(self.num_val_persons))
            self.logger.info('Num. of test persons: \t{}'.format(self.num_test_persons))
            self.logger.info('Num. of classes: \t\t{}'.format(self.num_classes))
            self.logger.info('Decode image shape: \t\t{}'.format(self.decode_img_shape))
            self.logger.info('Single img shape: \t\t{}'.format(self.single_img_shape))
            self.logger.info('Training TFrecord path: \t{}'.format(self.train_path))
            self.logger.info('Validation TFrecord path: \t{}'.format(self.val_path))
            self.logger.info('Test TFrecord path: \t\t{}'.format(self.test_path))
            self.logger.info('Overfitting TFrecord path: \t{}'.format(self.overfitting_path))


class OpenEDS_Identity(object):
    def __init__(self, name='OpenEDS', track='Identification', is_train=True, resized_factor=0.5, log_dir=None):
        self.name = name
        self.track = track

        self.num_train_imgs = 342419
        self.num_validation_imgs = 12759
        self.num_identities = 152
        self.decode_img_shape = (int(640 * resized_factor), int(400 * resized_factor), 3)

        # TFrecprd [atj
        self.train_path = '../../Data/OpenEDS/{}/train/train.tfrecords'.format(self.track)
        self.val_path = '../../Data/OpenEDS/{}/validation/validation.tfrecords'.format(self.track)

        if is_train:
            self.logger = logging.getLogger(__name__)   # logger
            self.logger.setLevel(logging.INFO)
            utils.init_logger(logger=self.logger, logDir=log_dir, isTrain=is_train, name='dataset')

            self.logger.info('Dataset name: \t\t{}'.format(self.name))
            self.logger.info('Dataset track: \t\t{}'.format(self.track))
            self.logger.info('Num. of training imgs: \t{}'.format(self.num_train_imgs))
            self.logger.info('Num. of validation imgs: \t{}'.format(self.num_validation_imgs))
            self.logger.info('Num. of identities: \t\t{}'.format(self.num_identities))
            self.logger.info('Decode mage shape: \t\t{}'.format(self.decode_img_shape))
            self.logger.info('Training TFrecord path: \t{}'.format(self.train_path))
            self.logger.info('Validation TFrecord path: \t{}'.format(self.val_path))

    def __call__(self):
        return self.train_path, self.val_path


class OpenEDS(object):
    def __init__(self, name='OpenEDS', track='Semantic_Segmentation_Dataset',
                 isTrain=True, resizedFactor=0.5, logDir=None):
        self.name = name
        self.track = track

        self.numTrainImgs = 8916
        self.numValImgs = 2403
        self.numTestImgs = 1440

        self.numTrainPersons = 95
        self.numValPersons = 28
        self.numTestPersons = 29
        self.numClasses = 4

        self.decodeImgShape = (int(640 * resizedFactor), int(400 * 2 * resizedFactor), 1)
        self.singleImgShape = (int(640 * resizedFactor), int(400 * resizedFactor), 1)

        # TFrecord path
        self.trainPath = '../../Data/OpenEDS/{}/train/train.tfrecords'.format(self.track)
        self.valPath = '../../Data/OpenEDS/{}/validation/validation.tfrecords'.format(self.track)
        self.testPath = '../../Data/OpenEDS/{}/test/test.tfrecords'.format(self.track)
        self.overfittingPath = '../../Data/OpenEDS/{}/overfitting/overfitting.tfrecords'.format(self.track)

        if isTrain:
            self.logger = logging.getLogger(__name__)   # logger
            self.logger.setLevel(logging.INFO)
            utils.init_logger(logger=self.logger, logDir=logDir, isTrain=isTrain, name='dataset')

            self.logger.info('Dataset name: \t\t{}'.format(self.name))
            self.logger.info('Dataset track: \t\t{}'.format(self.track))
            self.logger.info('Num. of training imgs: \t{}'.format(self.numTrainImgs))
            self.logger.info('Num. of validation imgs: \t{}'.format(self.numValImgs))
            self.logger.info('Num. of test imgs: \t\t{}'.format(self.numTestImgs))
            self.logger.info('Num. of training persons: \t{}'.format(self.numTrainPersons))
            self.logger.info('Num. of validation persons: \t{}'.format(self.numValPersons))
            self.logger.info('Num. of test persons: \t{}'.format(self.numTestPersons))
            self.logger.info('Num. of classes: \t\t{}'.format(self.numClasses))
            self.logger.info('Decode image shape: \t{}'.format(self.decodeImgShape))
            self.logger.info('Single img shape: \t\t{}'.format(self.singleImgShape))
            self.logger.info('Training TFrecord path: \t{}'.format(self.trainPath))
            self.logger.info('Validation TFrecord path: \t{}'.format(self.valPath))
            self.logger.info('Test TFrecord path: \t\t{}'.format(self.testPath))
            self.logger.info('Overfitting TFrecord path: \t\t{}'.format(self.overfittingPath))

    def __call__(self, isTrain=True):
        if isTrain:
            return self.trainPath, self.valPath, self.overfittingPath
        else:
            return self.testPath, self.valPath, None

def Dataset(name, track='Semantic_Segmentation_Dataset', isTrain=True, resizedFactor=0.5, logDir=None):
    if name == 'OpenEDS' and track == 'Semantic_Segmentation_Dataset':
        return OpenEDS(name=name, track=track, isTrain=isTrain, resizedFactor=resizedFactor, logDir=logDir)
    elif name == 'OpenEDS' and track == 'Identification':
        return OpenEDS_Identity(name=name, track=track, is_train=isTrain, resized_factor=resizedFactor, log_dir=logDir)
    elif name== 'OpenEDS' and track == 'Generative_Dataset':
        return OpenEDS_Generation(name=name, track=track, is_train=isTrain, resized_factor=resizedFactor, log_dir=logDir)
    else:
        raise NotImplementedError
