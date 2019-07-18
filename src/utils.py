import os
import json
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
        self.stage = stage

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


def all_files_under(folder, subfolder='images', endswith='.png'):
    new_folder = os.path.join(folder, subfolder)
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
