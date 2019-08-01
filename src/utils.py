# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import os
import json
import logging
import cv2
import numpy as np


class JsonData(object):
    # stage - 0 semantic_segmentation_images
    # stage - 1 generative_images
    def __init__(self, file_path, stage=0):
        file = open(file_path).read()
        self.json_data = json.loads(file)
        self.key = 'semantic_segmenation_images' if stage == 0 else "generative_images"

    def find_id(self, target):
        target = target.replace('.npy', '.png')

        for i, item in enumerate(self.json_data):
            for img_name in item[self.key]:
                if img_name == target:
                    user_id = item['id']
                    return user_id


class SSData(object):
    def __init__(self, data_path, json_path, stage):
        self.dataPath = os.path.join(data_path, stage)
        self.jsonPath = json_path
        self.stage = stage if stage != 'overfitting' else 'train'

        # Read image paths
        self.img_paths = all_files_under(self.dataPath, subfolder='images', endswith='.png')
        print('Number of images in img_paths: {}'.format(len(self.img_paths)))

        # Read label paths
        self.label_paths = None
        if self.stage.lower() != 'test':
            self.label_paths = all_files_under(self.dataPath, subfolder='labels', endswith='.npy')
            print('Number of labels in label_paths: {}'.format(len(self.label_paths)))

        # Read json file to find user ID
        json_file_path = os.path.join(self.jsonPath, 'OpenEDS_{}_userID_mapping_to_images.json'.format(self.stage))
        self.jsonDataObj = JsonData(file_path=json_file_path, stage=0)

    def back_info(self, imgPath, labelPath=None):
        # Find user ID
        userId = self.jsonDataObj.find_id(target=os.path.basename(imgPath))
        # Name of the image
        imgName = os.path.basename(imgPath)

        # Read img in grayscale
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        # Read .npy label
        if labelPath is None:
            label = np.zeros_like(img)
        else:
            label = np.load(labelPath)

        data = init_canvas(img.shape[0], img.shape[1], channel=1, img1=img, img2=label, times=2, axis=1)

        return data, userId, imgName


def make_folders(isTrain=True, curTime=None, subfolder=None):
    modelDir = os.path.join('model', subfolder, '{}'.format(curTime))
    logDir = os.path.join('log', subfolder, '{}'.format(curTime))
    sampleDir = os.path.join('sample', subfolder, '{}'.format(curTime))
    valDir, testDir = None, None

    if isTrain:
        if not os.path.isdir(modelDir):
            os.makedirs(modelDir)

        if not os.path.isdir(logDir):
            os.makedirs(logDir)

        if not os.path.isdir(sampleDir):
            os.makedirs(sampleDir)


    else:
        valDir = os.path.join('val', subfolder, '{}'.format(curTime))
        testDir = os.path.join('test', subfolder, '{}'.format(curTime))

        if not os.path.isdir(valDir):
            os.makedirs(valDir)

        if not os.path.isdir(testDir):
            os.makedirs(testDir)

    return modelDir, logDir, sampleDir, valDir, testDir


def init_logger(logger, logDir, name, isTrain):
    logger.propagate = False  # solve print log multiple times problems
    fileHandler, streamHandler = None, None

    if isTrain:
        formatter = logging.Formatter(' - %(message)s')

        # File handler
        fileHandler = logging.FileHandler(os.path.join(logDir, name + '.log'))
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.INFO)

        # Stream handler
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        # Add handlers
        if not logger.handlers:
            logger.addHandler(fileHandler)
            logger.addHandler(streamHandler)

    return logger, fileHandler, streamHandler


def print_main_parameters(logger, flags, isTrain=False):
    if isTrain:
        logger.info('gpu_index: \t\t\t{}'.format(flags.gpu_index))
        logger.info('dataset: \t\t\t{}'.format(flags.dataset))
        logger.info('method: \t\t\t{}'.format(flags.method))
        logger.info('multi_test: \t\t{}'.format(flags.multi_test))
        logger.info('batch_size: \t\t\t{}'.format(flags.batch_size))
        logger.info('resize_factor: \t\t{}'.format(flags.resize_factor))
        logger.info('is_train: \t\t\t{}'.format(flags.is_train))
        logger.info('learing_rate: \t\t{}'.format(flags.learning_rate))
        logger.info('weight_decay: \t\t{}'.format(flags.weight_decay))
        logger.info('iters: \t\t\t{}'.format(flags.iters))
        logger.info('print_freq: \t\t\t{}'.format(flags.print_freq))
        logger.info('sample_freq: \t\t{}'.format(flags.sample_freq))
        logger.info('eval_freq: \t\t\t{}'.format(flags.eval_freq))
        logger.info('load_model: \t\t\t{}'.format(flags.load_model))
    else:
        print('-- gpu_index: \t\t{}'.format(flags.gpu_index))
        print('-- dataset: \t\t{}'.format(flags.dataset))
        print('-- method: \t\t{}'.format(flags.method))
        print('-- multi_test: \t\t{}'.format(flags.multi_test))
        print('-- batch_size: \t\t{}'.format(flags.batch_size))
        print('-- resize_factor: \t\t{}'.format(flags.resize_factor))
        print('-- is_train: \t\t{}'.format(flags.is_train))
        print('-- learing_rate: \t{}'.format(flags.learning_rate))
        print('-- weight_decay: \t{}'.format(flags.weight_decay))
        print('-- iters: \t\t{}'.format(flags.iters))
        print('-- print_freq: \t\t{}'.format(flags.print_freq))
        print('-- sample_freq: \t{}'.format(flags.sample_freq))
        print('-- eval_freq: \t\t{}'.format(flags.eval_freq))
        print('-- load_model: \t\t{}'.format(flags.load_model))



def all_files_under(folder, subfolder='images', endswith='.png'):
    if subfolder is not None:
        new_folder = os.path.join(folder, subfolder)
    else:
        new_folder = folder

    file_names =  [os.path.join(new_folder, fname)
                   for fname in os.listdir(new_folder) if fname.endswith(endswith)]

    return sorted(file_names)


def init_canvas(h, w, channel, img1, img2, times=1, axis=0):
    canvas = None
    if axis==0:
        canvas = np.squeeze(np.zeros((times * h,  w, channel), dtype=np.uint8))
        canvas[:h, :] = img1
        canvas[h:, :] = img2
    elif axis==1:
        canvas = np.squeeze(np.zeros((h, times * w, channel), dtype=np.uint8))
        canvas[:, :w] = img1
        canvas[:, w:] = img2

    return canvas


def convert_color_label(img):
    yellow = [102, 255, 255]
    green = [102, 204, 0]
    cyan = [153, 153, 0]
    violet = [102, 0, 102]

    # 0: background - violet
    # 1: sclera - cyan
    # 2: iris - green
    # 3: pupil - yellow
    img_rgb = np.zeros([*img.shape, 3], dtype=np.uint8)
    for i, color in enumerate([violet, cyan, green, yellow]):
        img_rgb[img == i] = color

    return img_rgb


def save_npy(data, save_dir, file_name, size=(640, 400)):
    save_dir = os.path.join(save_dir, 'npy')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Extract image number from [000002342342_U23.png]
    file_name = file_name[0].split('_')[0]

    # Convert [1, H, W] to [H, W]
    data = np.squeeze(data)

    # Resize from [H/2, W/2] to [H, W]
    data = cv2.resize(data, dsize=size, interpolation=cv2.INTER_NEAREST)

    # Convert data type from int32 to uint8
    data = data.astype(np.uint8)

    # Save data in npy format by requirement
    np.save(os.path.join(save_dir, file_name), data)


def save_imgs(img_stores, iterTime=None, saveDir=None, margin=5, img_name=None, name_append='', is_vertical=True):
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)

    num_categories = len(img_stores)
    for i in range(num_categories):
        if img_stores[i].shape[-1] == 1:
            img_stores[i] = np.squeeze(img_stores[i], axis=-1)

        img_stores[i] = img_stores[i].astype(np.uint8)

    numImgs, h, w = img_stores[0].shape

    if is_vertical:
        canvas = np.zeros((num_categories * h + (num_categories + 1) * margin,
                           numImgs * w + (numImgs + 1) * margin, 3), dtype=np.uint8)

        for i in range(numImgs):
            for j in range(num_categories):
                if j != 0:  # label map
                    canvas[(j+1)*margin+j*h:(j+1)*margin+(j+1)*h, (i+1)*margin+i*w:(i+1)*(margin+w), :] = \
                        convert_color_label(img_stores[j][i])
                else:
                    canvas[(j+1)*margin+j*h:(j+1)*margin+(j+1)*h, (i+1)*margin+i*w:(i+1)*(margin+w), :] = \
                        np.dstack((img_stores[j][i], img_stores[j][i], img_stores[j][i]))

    else:
        canvas = np.zeros((numImgs * h + (numImgs + 1) * margin,
                           num_categories * w + (num_categories + 1) * margin, 3), dtype=np.uint8)

        for i in range(numImgs):
            for j in range(num_categories):
                if j != 0:
                    canvas[(i+1)*margin+i*h:(i+1)*(margin+h), (j+1)*margin+j*w:(j+1)*margin+(j+1)*w, :] = \
                        convert_color_label(img_stores[j][i])
                else:
                    canvas[(i+1)*margin+i*h:(i+1)*(margin+h), (j+1)*margin+j*w:(j+1)*margin+(j+1)*w, :] = \
                        np.dstack((img_stores[j][i], img_stores[j][i], img_stores[j][i]))

    if img_name is None:
        cv2.imwrite(os.path.join(saveDir, str(iterTime).zfill(6) + '.png'), canvas)
    else:
        cv2.imwrite(os.path.join(saveDir, name_append+img_name[0]), canvas)
