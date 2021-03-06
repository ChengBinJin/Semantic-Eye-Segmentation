# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import os
import random
import csv
import json
import logging
import cv2
import numpy as np
from scipy.ndimage import rotate


class ImagePool(object):
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.imgs = list()

    def query(self, img):
        if self.pool_size == 0:
            return img

        if len(self.imgs) < self.pool_size:
            self.imgs.append(img)
            return img
        else:
            if random.random() > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp_img = self.imgs[random_id].copy()
                self.imgs[random_id] = img.copy()
                return tmp_img
            else:
                return img


class JsonData(object):
    def __init__(self, is_statistics=False):
        path = "../../Data/OpenEDS/OpenEDS_{}_userID_mapping_to_images.json"
        self.keys = ["semantic_segmentation_images", "generative_images", "sequence_images"]

        train_file = open(path.format("train")).read()
        val_file = open(path.format("validation")).read()
        test_file = open(path.format("test")).read()

        self.train_json_data = json.loads(train_file)
        self.val_json_data = json.loads(val_file)
        self.test_json_data = json.loads(test_file)

        self._collect_user_id()

        if is_statistics:
            self.statistics()

    def find_id(self, target, data_set="train", key="semantic_segmentation_images"):
        target = target.replace('.npy', '.png')

        if data_set == "train":
            json_data = self.train_json_data
        elif data_set == "validation":
            json_data = self.val_json_data
        elif data_set == "test":
            json_data = self.test_json_data
        elif data_set == "overfitting":
            json_data = self.train_json_data
        else:
            raise NotImplementedError

        for i, item in enumerate(json_data):
            for img_name in item[key]:
                if img_name == target:
                    user_id = item['id']
                    return True, user_id

        return False, None

    def _collect_user_id(self):
        users_list = list()
        index = 0

        for i, json_data in enumerate([self.train_json_data, self.val_json_data, self.test_json_data]):
            for item in json_data:
                user = dict()

                user['id'] = item['id']
                user['state'] = ["train", "validation", "test"][i]
                user['cls'] = index

                if i != 2:
                    user['num_of_imgs'] = len(item[self.keys[0]]) + len(item[self.keys[1]]) + len(item[self.keys[2]])
                else:  # in "generative_images" there are no images
                    user['num_of_imgs'] = len(item[self.keys[0]]) + len(item[self.keys[2]])

                index += 1

                users_list.append(user)

        self.users_list = sorted(users_list, key=lambda k: k['id'])

    def get_user_ids(self):
        return self.users_list

    def statistics(self):
        # Initialize csv file
        f = open('../statistics/User_infor.csv', 'w', encoding='utf-8', newline='')
        writer = csv.writer(f)
        # Write tag info
        writer.writerow(['id', 'cls', 'num_of_imgs', 'state'])

        # Writ to csv file and print info
        total_imgs = 0
        for user in self.users_list:
            # Write
            writer.writerow([user['id'], int(user['cls']), int(user['num_of_imgs']), user['state']])

            msg = 'ID: {}, Index: {:3d}, Num: {:4d},  State: {}'
            print(msg.format(user['id'], user['cls'], user['num_of_imgs'], user['state']))

            total_imgs += user['num_of_imgs']

        f.close()
        print("Total imgs: {}".format(total_imgs))


class SSData(object):
    def __init__(self, data_path, json_path, stage):
        self.dataPath = os.path.join(data_path, stage)
        self.jsonPath = json_path
        self.stage = stage if stage != 'overfitting' else 'train'

        # Read image paths
        self.img_paths = all_files_under(self.dataPath, subfolder='images', endswith='.png')
        print('Number of images in img_paths: {}'.format(len(self.img_paths)))

        # Read label paths
        self.label_paths = all_files_under(self.dataPath, subfolder='labels', endswith='.npy')
        print('Number of labels in label_paths: {}'.format(len(self.label_paths)))

        # Read json file to find user ID
        # json_file_path = os.path.join(self.jsonPath, 'OpenEDS_{}_userID_mapping_to_images.json'.format(self.stage))
        self.jsonDataObj = JsonData()

    def back_info(self, imgPath, labelPath=None, stage='train'):
        # Find user ID
        flage, userId = self.jsonDataObj.find_id(target=os.path.basename(imgPath), data_set=stage)
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
    modelDir = os.path.join('../model', subfolder, '{}'.format(curTime))
    logDir = os.path.join('../log', subfolder, '{}'.format(curTime))
    sampleDir = os.path.join('../sample', subfolder, '{}'.format(curTime))
    valDir, testDir = None, None

    if isTrain:
        if not os.path.isdir(modelDir):
            os.makedirs(modelDir)

        if not os.path.isdir(logDir):
            os.makedirs(logDir)

        if not os.path.isdir(sampleDir):
            os.makedirs(sampleDir)
    else:
        valDir = os.path.join('../val', subfolder, '{}'.format(curTime))
        testDir = os.path.join('../test', subfolder, '{}'.format(curTime))

        if not os.path.isdir(valDir):
            os.makedirs(valDir)

        if not os.path.isdir(testDir):
            os.makedirs(testDir)

    return modelDir, logDir, sampleDir, valDir, testDir


def make_folders_simple(is_train=True, cur_time=None, subfolder=None):
    model_dir = os.path.join('../model', subfolder, '{}'.format(cur_time))
    log_dir = os.path.join('../log', subfolder, '{}'.format(cur_time))

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    return model_dir, log_dir

def init_logger(logger, logDir, name, isTrain):
    logger.propagate = False  # solve print log multiple times problem
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
        logger.info('advanced_multi_test: {}'.format(flags.advanced_multi_test))
        logger.info('batch_size: \t\t\t{}'.format(flags.batch_size))
        logger.info('resize_factor: \t{}'.format(flags.resize_factor))
        logger.info('use_dice_loss: \t{}'.format(flags.use_dice_loss))
        logger.info('use_batch_norm: \t{}'.format(flags.use_batch_norm))
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
        print('-- advanced_multi_test: {}'.format(flags.advanced_multi_test))
        print('-- batch_size: \t\t{}'.format(flags.batch_size))
        print('-- resize_factor: \t{}'.format(flags.resize_factor))
        print('-- use_dice_loss: \t{}'.format(flags.use_dice_loss))
        print('-- use_batch_norm: \t{}'.format(flags.use_batch_norm))
        print('-- is_train: \t\t{}'.format(flags.is_train))
        print('-- learing_rate: \t{}'.format(flags.learning_rate))
        print('-- weight_decay: \t{}'.format(flags.weight_decay))
        print('-- iters: \t\t{}'.format(flags.iters))
        print('-- print_freq: \t\t{}'.format(flags.print_freq))
        print('-- sample_freq: \t{}'.format(flags.sample_freq))
        print('-- eval_freq: \t\t{}'.format(flags.eval_freq))
        print('-- load_model: \t\t{}'.format(flags.load_model))


def all_files_under(folder, subfolder=None, endswith='.png'):
    if subfolder is not None:
        new_folder = os.path.join(folder, subfolder)
    else:
        new_folder = folder

    if os.path.isdir(new_folder):
        file_names =  [os.path.join(new_folder, fname)
                       for fname in os.listdir(new_folder) if fname.endswith(endswith)]
        return sorted(file_names)
    else:
        return []


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
    if data.shape[0:2] != size:
        data = cv2.resize(data, dsize=(size[1], size[0]), interpolation=cv2.INTER_NEAREST)

    # Convert data type from int32 to uint8
    data = data.astype(np.uint8)

    # Save data in npy format by requirement
    np.save(os.path.join(save_dir, file_name), data)


def inverse_cropping(imgs, resize_ratio=1.1, batch=11, num_parts=6, is_label=False, num_classes=4):
    num_imgs, h, w = imgs.shape[0:3]
    if is_label:
        new_shape = (batch, int(resize_ratio * h), int(resize_ratio * w), num_classes)
    else:
        new_shape = (batch, int(resize_ratio * h), int(resize_ratio * w), 1)

    h_margin = new_shape[1] - h
    w_margin = new_shape[2] - w

    results = list()
    for i, p in enumerate(range(0, num_imgs, batch)):
        canvas = np.zeros(new_shape)
        resized_canvas = np.zeros((batch, h, w, new_shape[3]))

        part_imgs = imgs[p:p+batch]
        if np.ndim(part_imgs) == 3:
            part_imgs = np.expand_dims(part_imgs, axis=3)

        if i % num_parts == 0:      # original
            results.extend(part_imgs)
            continue
        elif i % num_parts == 1:    # top-left
            canvas[:, :h, :w, :] = part_imgs
        elif i % num_parts == 2:    # top-right
            canvas[:, :h, w_margin:, :] = part_imgs
        elif i % num_parts == 3:    # center
            canvas[:, int(0.5*h_margin):int(0.5*h_margin)+h, int(0.5*w_margin):int(0.5*w_margin)+w, :] = part_imgs
        elif i % num_parts == 4:    # bottom-left
            canvas[:, h_margin:, :w, :] = part_imgs
        elif i % num_parts == 5:    # bottom-right
            canvas[:, h_margin:, w_margin:, :] = part_imgs

        for j in range(batch):
            res_img = cv2.resize(canvas[j], (w, h), cv2.INTER_LINEAR)
            if np.ndim(res_img) == 2:
                res_img = np.expand_dims(res_img, axis=2)
            resized_canvas[j] = res_img

        results.extend(resized_canvas)

    return np.asarray(results)


def inverse_flip(imgs):
    num_imgs = imgs.shape[0]
    num_half = int(0.5 * num_imgs)

    results = list()
    for i in range(num_imgs):
        img = np.squeeze(imgs[i].copy())

        if i >= num_half:
            # Vertical-axis flipping
            img = cv2.flip(img, flipCode=1)

        results.extend([img])

    return np.asarray(results)


def inverse_rotate(imgs, max_degree=10, interval=2, batch=11, is_label=False):
    num_imgs = imgs.shape[0]

    results = list()
    for i in range(0, num_imgs, batch):
        part_imgs = imgs[i:i+batch].copy()

        for idx, degree in enumerate(range(max_degree, -max_degree - 1, -interval)):
            if is_label:
                img_rotate = rotate(input=part_imgs[idx], angle=degree, axes=(0, 1), reshape=False, order=3,
                                    mode='constant', cval=0.)
            else:
                img_rotate = rotate(input=part_imgs[idx], angle=degree, axes=(0, 1), reshape=False, order=3,
                                mode='constant', cval=0.)
                img_rotate = np.clip(img_rotate, a_min=0., a_max=255.)

            results.extend([img_rotate])

    return np.asarray(results)

def save_imgs_indiv(imgs, w_num_imgs, save_dir=None, img_name=None, name_append='', is_label=False, margin=5,
                    factor=1.0):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if is_label:
        imgs = np.argmax(imgs, axis=3)
    else:
        imgs = np.squeeze(imgs).astype(np.uint8)

    num_imgs, h, w = imgs.shape[0:3]
    h_num_imgs = int(num_imgs // w_num_imgs)

    if is_label:
        canvas = np.zeros((h_num_imgs * h + (1 + h_num_imgs) * margin, w_num_imgs * w + (w_num_imgs + 1) * margin, 3),
                          dtype=np.uint8)
    else:
        canvas = np.zeros((h_num_imgs * h + (1 + h_num_imgs) * margin, w_num_imgs * w + (w_num_imgs + 1) * margin),
                          dtype=np.uint8)


    for j in range(num_imgs):
        x_idx = j // w_num_imgs
        y_idx = j % w_num_imgs

        if is_label:
            canvas[(x_idx + 1) * margin + x_idx * h:(x_idx + 1) * margin + (x_idx + 1) * h,
            (y_idx + 1) * margin + y_idx * w:(y_idx + 1) * margin + (y_idx + 1) * w] = convert_color_label(imgs[j])
        else:
            canvas[(x_idx + 1) * margin + x_idx * h:(x_idx + 1) * margin + (x_idx + 1) * h,
            (y_idx + 1) * margin + y_idx * w:(y_idx + 1) * margin + (y_idx + 1) * w] = imgs[j]

    canvas = cv2.resize(canvas, None, fx=factor, fy=factor)
    cv2.imwrite(os.path.join(save_dir, name_append + img_name[0]), canvas)

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
