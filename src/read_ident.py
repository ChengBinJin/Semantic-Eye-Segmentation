# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Synthetic Eye Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------

import os
import utils as utils
from utils import JsonData


def get_ident_data(state="validation"):
    if state.lower() == "train":
        paths = ["../../Data/OpenEDS/Generative_Dataset", "../../Data/OpenEDS/Sequence_Dataset"]
    elif state.lower() == "validation":
        paths = ["../../Data/OpenEDS/Semantic_Segmentation_Dataset"]
    else:
        raise NotImplementedError

    # Initilize JsonData to read all of the information from the json files
    json_obj = JsonData(is_statistics=False)

    full_img_paths = list()
    full_cls = list()

    num_imgs = 0
    for path in paths:
        for idx, (root, directories, files) in enumerate(os.walk(path)):
            for directory in directories:
                folder = os.path.join(root, directory)
                img_paths = utils.all_files_under(folder, subfolder=None, endswith='.png')

                if (len(img_paths) != 0) & ('paired' not in folder) & ('overfitting' not in folder) & \
                        ('train_expand' not in folder):
                    # Add img paths
                    full_img_paths.extend(img_paths)

                    data_set = os.path.basename(os.path.dirname(folder))
                    key = os.path.basename(path).lower().replace('dataset', 'images')
                    for img_path in img_paths:
                        # Read user id
                        flag, user_id = json_obj.find_id(target=os.path.basename(img_path),
                                                         data_set=data_set,
                                                         key=key)
                        for item in json_obj.users_list:
                            if item['id'] == user_id:
                                # Add cls info of the image
                                full_cls.extend([item['cls']])

                        if not flag:
                            exit("Cannot find user id of the image {} !".format(img_path))

                    num_imgs += len(img_paths)
                    print('The number of colllected data: {}'.format(num_imgs))

    print("Total data: {}".format(num_imgs))

    return full_img_paths, full_cls


if __name__ == '__main__':
    # state == 0 for training data
    # state == 1 for validation data

    get_ident_data(state="train")
