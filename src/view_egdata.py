# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Synthetic Eye Generation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# --------------------------------------------------------------------------

import csv
from utils import JsonData

def main(paths):
    # num_imgs = 0
    #
    # for path in paths:
    #     for idx, (root, directories, files) in enumerate(os.walk(path)):
    #         for directory in directories:
    #             folder = os.path.join(root, directory)
    #             img_paths = utils.all_files_under(folder, subfolder=None, endswith='.png')
    #
    #             if (len(img_paths) != 0) & ('paired' not in folder) & ('overfitting' not in folder):
    #                 # print('path: {}'.format(folder))
    #                 # print('num of imgs: {}'.format(len(img_paths)))
    #
    #                 # for img_path in img_paths:
    #                 #     print(img_path)
    #
    #                 num_imgs += len(img_paths)
    #
    # print("Total images: {}".format(num_imgs))

    # Initilize JsonData to read all of the information from the json files
    json_obj = JsonData()
    users = json_obj.get_user_ids()

    # Initialize csv file
    f = open('../statistics/User_infor.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(f)
    # Write tag info
    writer.writerow(['id', 'index', 'num_of_imgs', 'state'])

    # Writ to csv file and print info
    total_imgs = 0
    for user in users:
        # Write
        writer.writerow([user['id'], int(user['index']), int(user['num_of_imgs']), user['state']])

        msg = 'ID: {}, Index: {:3d}, Num: {:4d},  State: {}'
        print(msg.format(user['id'], user['index'], user['num_of_imgs'], user['state']))

        total_imgs += user['num_of_imgs']

    f.close()
    print("Total imgs: {}".format(total_imgs))


if __name__ == '__main__':
    # paths = ["../../Data/OpenEDS/Generative_Dataset", "../../Data/OpenEDS/Semantic_Segmentation_Dataset",
    #          "../../Data/OpenEDS/Sequence_Dataset"]

    train_path = ["../../Data/OpenEDS/Generative_Dataset", "../../Data/OpenEDS/Sequence_Dataset"]
    val_path = ["../../Data/OpenEDS/Semantic_Segmentation_Dataset"]

    main(train_path)
