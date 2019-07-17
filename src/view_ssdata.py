import os
import cv2
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', dest='data_path', default='../../Data/OpenEDS/Semantic_Segmentation_Dataset',
                    help='OpenEDS semantic segmentation data path')
parser.add_argument('--stage', dest='stage', default='train',
                    help='Select one of the stage in [train|validation|test]')
args = parser.parse_args()


def main():
    dataPath = os.path.join(args.data_path, args.stage)

    img_paths = all_files_under(dataPath, subfolder='images', endswith='.png')
    print('Number of images in img_paths: {}'.format(len(img_paths)))

    label_paths = None
    if args.stage.lower() != 'test':
        label_paths = all_files_under(dataPath, subfolder='labels', endswith='.npy')
        print('Number of labels in label_paths: {}'.format(len(label_paths)))

    if args.stage.lower() != 'test':
        show_image(img_paths, label_paths)
    else:
        show_image(img_paths)


def show_image(imgPaths, labelPaths=None, hPos=10, wPos=10):
    for i, imgPath in enumerate(imgPaths):
        # Window name
        winName = os.path.basename(imgPath.replace('.png', ''))

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

        cv2.imshow(winName, canvas)
        if cv2.waitKey(0) & 0xff == 27:
            exit('Esc clicked')

        cv2.destroyWindow(winName)


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


def all_files_under(folder, subfolder='images', endswith='.png'):
    new_folder = os.path.join(folder, subfolder)
    file_names =  [os.path.join(new_folder, fname)
                   for fname in os.listdir(new_folder) if fname.endswith(endswith)]

    return sorted(file_names)

if __name__ == '__main__':
    main()
