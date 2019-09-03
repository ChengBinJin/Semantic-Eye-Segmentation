import os
import argparse
from utils import JsonData, all_files_under

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path_list', dest='data_path_list', type=str, default='input_folder_list.txt',
                    help='input data folder addresses are listed in the txt file')
args = parser.parse_args()

def main(data_path_list):
    img_paths, user_ids = read_data(data_path_list)
    print(len(img_paths))
    print(len(user_ids))


def read_data(data_path_list):
    file = open(data_path_list, 'r')
    paths = file.readlines()
    json_obj = JsonData()

    overall_paths = list()
    overall_user_id = list()
    for i, path in enumerate(paths):
        stage = os.path.dirname(path).split('/')[-1]

        img_paths = all_files_under(folder=path, subfolder=None, endswith='.png')
        overall_paths.extend(img_paths)
        print('{}: {} - num. of images: {}'.format(i+1, path, len(img_paths)))

        for j, img_path in enumerate(img_paths):
            _, user_id = json_obj.find_id(target=os.path.basename(img_path),
                                          data_set=stage,
                                          key='generative_images')

            overall_user_id.extend([user_id])

            if j % 500 == 0:
                print('Processed {} / {}'.format(j, len(img_paths)))

    return overall_paths, overall_user_id


if __name__ == '__main__':
    main(data_path_list=args.data_path_list)


