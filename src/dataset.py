# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import logging
import utils as utils

class OpenEDS(object):
    def __init__(self, name='OpenEDS', track='Semantic_Segmentation_Dataset',
                 isTrain=True, resizedFactor=0.5, logDir=None):
        self.name = name
        self.track = track

        self.numTrainImgs = 8916
        self.numValImgs = 2430
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

    def __call__(self, isTrain=True):
        if isTrain:
            return self.trainPath, self.valPath
        else:
            return self.testPath, None

def Dataset(name, track='Semantic_Segmentation_Dataset', isTrain=True, resizedFactor=0.5, logDir=None):
    if name == 'OpenEDS':
        return OpenEDS(name=name, track=track, isTrain=isTrain, resizedFactor=resizedFactor, logDir=logDir)
    else:
        raise NotImplementedError
