# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import cv2
import argparse

from utils import SSData


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str,
                    default='../../Data/OpenEDS/Semantic_Segmentation_Dataset',
                    help='OpenEDS semantic segmentation data path')
parser.add_argument('--json_path', dest='json_path', type=str,
                    default='../../Data/OpenEDS',
                    help='json file path which includes user ID')
parser.add_argument('--stage', dest='stage', type=str,
                    default='train',
                    help='Select one of the stage in [train|validation|test|overfitting]')
args = parser.parse_args()


def main(dataPath, jsonPath, stage):
    ssDataObj = SSData(dataPath, jsonPath, stage)
    numImgs = len(ssDataObj.img_paths)

    for i, imgPath in enumerate(ssDataObj.img_paths):
        if i % 200 == 0:
            print('Processing {} / {}...'.format(i, numImgs))

        labelPath = ssDataObj.label_paths[i]
        canvas, userId, imgName = ssDataObj.back_info(imgPath, labelPath, stage=stage)
        save_image(img=canvas, imgName=imgName.replace('.png', '') + '_' + userId,
                   folder=os.path.join(dataPath, stage, 'paired'))


def save_image(img, imgName, folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    cv2.imwrite(os.path.join(folder, imgName + '.png'), img)


if __name__ == '__main__':
    main(args.data_path, args.json_path, args.stage)


