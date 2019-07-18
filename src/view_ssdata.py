# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------
import os
import cv2
import argparse
import numpy as np

import utils as utils
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
                    help='Select one of the stage in [train|validation|test]')
parser.add_argument('--delay', dest='delay', type=int,
                    default=1000,
                    help='time delay when showing image')
parser.add_argument('--hide_img', dest='hide_img', action='store_true',
                    default=False,
                    help='show image or not')
parser.add_argument('--save_img', dest='save_img', action='store_true',
                    default=False,
                    help='save image in debegImgs folder or not')
args = parser.parse_args()


def show_image(ssData, hPos=10, wPos=10, saveFolder='../debugImgs'):
    preUserId = None
    for i, imgPath in enumerate(ssData.img_paths):
        if i % 200 == 0:
            print('Iteration: {:4d}'.format(i))

        # Read user ID
        userId = ssData.jsonDataObj.find_id(target=os.path.basename(imgPath))

        winName = None
        if not args.hide_img:
            # Window name
            winName = os.path.basename(imgPath.replace('.png', '')) + ' - ' + userId

            # Initialize window and move to the fixed display position
            cv2.namedWindow(winName)
            cv2.moveWindow(winName, wPos, hPos)

        # Read image and load npy file
        img = cv2.imread(imgPath)

        labelBgr = np.zeros_like(img)
        if ssData.label_paths is not None:
            label = np.load(ssData.label_paths[i])
            # label 0~3 data convert to BGR [0~255, 0~255, 0~255] data
            labelBgr = utils.convert_color_label(label)

        # Intilize canvas and copy the images
        h, w, c = img.shape
        canvas = utils.init_canvas(h, w, c, img1=img, img2=labelBgr, times=2, axis=1)

        # Show image
        if not args.hide_img:
            cv2.imshow(winName, canvas)
            if cv2.waitKey(args.delay) & 0xff == 27:
                exit('Esc clicked')

            # Delete all defined window
            cv2.destroyWindow(winName)

        # Save first image of the each user
        if args.save_img and (preUserId is not userId):
            save_image(img=canvas, imgName=args.stage + '_' + userId, folder=saveFolder)

        preUserId = userId


def save_image(img, imgName, folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    cv2.imwrite(os.path.join(folder, imgName + '.png'), img)


if __name__ == '__main__':
    ssDataObj = SSData(args.data_path, args.json_path, args.stage)
    show_image(ssDataObj)
