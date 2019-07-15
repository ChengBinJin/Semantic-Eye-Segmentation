import os
import cv2
import numpy as np

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


def show_img(img, name):
    # Move window to the specific fixed position
    cv2.namedWindow(name)
    cv2.moveWindow(name, 10, 10)

    cv2.imshow(name, img)
    if cv2.waitKey(0) & 0xff == 27:
        exit("Esc clicked")

    cv2.destroyWindow(name)


def read_labels(path):
    filenames = [[os.path.join(path, fname), os.path.basename(fname)]
                 for fname in os.listdir(path) if fname.endswith('.npy')]

    for filename, basename in filenames:
        label = np.load(filename)

        # Convert [0~3] labels to [0~255, 0~255, 0~255] BGR images
        label_rgb = convert_color_label(label)

        # Show image
        show_img(img=label_rgb, name=basename)


if __name__ == '__main__':
    test_label_path = '../../Data/OpenEDS/Generative_Dataset/test/labels/'
    read_labels(test_label_path)
