# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import json
import cv2
import numpy as np


class JsonDataObj(object):
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


def show_img(img, name, user_id=None):
    full_name = name + ' - ' + user_id

    # Move window to the specific fixed position
    cv2.namedWindow(full_name)
    cv2.moveWindow(full_name, 10, 10)

    cv2.imshow(full_name, img)
    if cv2.waitKey(0) & 0xff == 27:
        exit("Esc clicked")

    cv2.destroyWindow(full_name)


def read_labels(img_path, json_path):
    filenames = [[os.path.join(img_path, fname), os.path.basename(fname)]
                 for fname in os.listdir(img_path) if fname.endswith('.npy')]

    jsonObj = JsonDataObj(json_path, stage=1)

    for filename, basename in filenames:
        label = np.load(filename)

        # Convert [0~3] labels to [0~255, 0~255, 0~255] BGR images
        label_rgb = convert_color_label(label)
        # Find user id
        user_id = jsonObj.find_id(target=basename)
        # Show image
        show_img(img=label_rgb, name=basename, user_id=user_id)


if __name__ == '__main__':
    test_label_path = '../../Data/OpenEDS/Generative_Dataset/test/labels/'
    json_file_path = '../../Data/OpenEDS/OpenEDS_test_userID_mapping_to_images.json'
    read_labels(test_label_path, json_file_path)

