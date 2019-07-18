import os
import json
import cv2
import argparse
import numpy as np

import utils as utils


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', type=str,
                    default='../../Data/OpenEDS/Semantic_Segmentation_Dataset',
                    help='OpenEDS semantic segmentation data path')
parser.add_argument('--json_path', dest='json_path', type=str,
                    default='../../Data/OpenEDS',
                    help='json file path whithc includes user ID')
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


def main():
    dataPath = os.path.join(args.data_path, args.stage)

    img_paths = utils.all_files_under(dataPath, subfolder='images', endswith='.png')
    print('Number of images in img_paths: {}'.format(len(img_paths)))

    label_paths = None
    if args.stage.lower() != 'test':
        label_paths = utils.all_files_under(dataPath, subfolder='labels', endswith='.npy')
        print('Number of labels in label_paths: {}'.format(len(label_paths)))

    # Read json file to find user ID
    json_file_path = os.path.join(args.json_path, 'OpenEDS_{}_userID_mapping_to_images.json'.format(args.stage))
    jsonDataObj = JsonData(file_path=json_file_path, stage=0)

    if args.stage.lower() != 'test':
        show_image(img_paths, labelPaths=label_paths, jsonObj=jsonDataObj)
    else:
        show_image(img_paths, jsonObj=jsonDataObj)


def show_image(imgPaths, labelPaths=None, jsonObj=None, hPos=10, wPos=10, saveFolder='../debugImgs'):
    preUserId = None
    for i, imgPath in enumerate(imgPaths):
        if i % 200 == 0:
            print('Iteration: {:4d}'.format(i))

        # Read user ID
        userId = jsonObj.find_id(target=os.path.basename(imgPath))

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
        if labelPaths is not None:
            label = np.load(labelPaths[i])
            # label 0~3 data convert to BGR [0~255, 0~255, 0~255] data
            labelBgr = convert_color_label(label)

        # Intilize canvas and copy the images
        h, w, c = img.shape
        canvas = init_canvas(h, w, c, img1=img, img2=labelBgr, times=2, axis=1)

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


def init_canvas(h, w, c, img1, img2, times=1, axis=0):
    canvas = None
    if axis==0:
        canvas = np.zeros((times * h,  w, c), dtype=np.uint8)
        canvas[:h, :] = img1
        canvas[h:, :] = img2
    elif axis==1:
        canvas = np.zeros((h, times * w, c), dtype=np.uint8)
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


if __name__ == '__main__':
    main()
