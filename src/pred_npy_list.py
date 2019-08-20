# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import os
import cv2
import argparse
import numpy as np
import utils as utils


parser = argparse.ArgumentParser(description='make a full path list for the prediciton results')
parser.add_argument('--method', dest='method', type=str, default='U-Net-light-v4_2',
                    help='select a method from the list [U-Net, U-Net-light-v1, U-Net-light-v2, U-Net-light-v3, '
                         'U-Net-light-v4, U-Net-light-v4_1]')
parser.add_argument('--model', dest='model', type=str, default='20190725-161826',
                    help='path pointing predictions of the .npy files')
parser.add_argument('--show_img', dest='show_img', action='store_true', default=False,
                    help='show label image for debugging')
parser.add_argument('--delay', dest='delay', type=int, default=0, help='time delay when showing image')
args = parser.parse_args()


def main(method, model, delay, show_img):
    # Check npy folder exist
    path = os.path.join('../test', method, model, 'npy')
    if not os.path.isdir(path):
        exit("Cannot find folder {}...".format(path))

    # Make saving-folder of the submit/load_model/pred_npy_list.txt
    save_path = os.path.join('../submit', method, model)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Initialize txt file and open it
    txt_file = open(os.path.join(save_path, 'pred_npy_list.txt'), 'w')

    # Read all files in the folder
    file_names = utils.all_files_under(folder=path, subfolder=None, endswith='.npy')

    print(" [*] Writing the full path of the npy file ...")
    for file_name in file_names:
        full_path = os.path.abspath(file_name)
        txt_file.write(full_path + '\n')

        # Show image
        if show_img:
            img = np.load(file_name)  # read npy file

            cv2.imshow('Show', utils.convert_color_label(img))
            if cv2.waitKey(delay) & 0xFF == 27:
                exit("Esc clicked!")

    print(" [!] Finished to write!")

    # Close txt file
    txt_file.close()


if __name__ == '__main__':
    main(args.method, args.model, args.delay, args.show_img)
