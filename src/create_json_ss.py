import os
import sys
import ntpath
import base64
import numpy as np
import json
import argparse


def np_to_base64_utf8_str(arr):
    np_buff = arr.tobytes()
    np_buff_b64_bytes = base64.b64encode(np_buff)
    np_buff_base64_utf8_string = np_buff_b64_bytes.decode('utf-8')
    
    return np_buff_base64_utf8_string

def create_json_from_npy_data(dataset_filename, json_filename, num_model_params=-1):
    """
    @param::num_model_params: Number of model parameters
    @param::dataset_filename: A text file with each line containing the full path to label files.
                              Every label file is .npy extension file that contains the labels
                              as a 2D numpy array of data-type uint8.
    @param::json_filename: The filepath to the json file that you can submit
    :return:
    """

    # I am assuming all files to have .npy extension
    # print(dataset_filename)
    if not os.path.isfile(dataset_filename):
        raise ("Label file list does not exist")
        
    with open(dataset_filename,'r') as f:
        lines = f.readlines()
        lines = [l.rstrip() for l in lines]
    
    data = {}
    data['num_model_params'] = num_model_params
    data['number_of_samples'] = len(lines)
    data['labels'] = {}

    for idx, l in enumerate(lines):
        if idx % 100 == 0:
            print("Saving {}...".format(idx))

        if not os.path.isfile(l):
            raise("File does not exist: {}".format(l))
            
        temp_label = np.load(l)

        # save as uint8
        temp_label = temp_label.astype(np.uint8)
        np_b64_utf8_str = np_to_base64_utf8_str(temp_label)
        data['labels'][ntpath.basename(l[:-4])] = np_b64_utf8_str

    with open(json_filename, 'w') as f:
        json.dump(data,f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to create submission JSON file for OpenEDS competition held on EvalAI platform.")
    parser.add_argument("--method", type=str, dest="method",
                        help="Select a method from [U-Net, U-Net-light-v1, U-Net-light-v2, U-Net-light-v3]",
                        default="U-Net-light-v3")
    parser.add_argument("--list-file", type=str, dest="list_file",
                        help="Path to text file containing list of FULL FILEPATH of labels saved in .npy format", 
                        default=None)
    parser.add_argument("--submission-json", type=str, dest="submission_json",
                        help="Path to JSON filename for submission.", default="submission")
    parser.add_argument("--num-model-param", type=int, dest="num_model_param",
                        help="Number of model parameters for submission.", default=-1)

    args = parser.parse_args()

    if args.list_file is None:
        print('Enter the filepath with label list, quitting...')
        sys.exit(0)
    
    if args.submission_json is None:
        print('Enter the submission JSON filepath, quitting...')
        sys.exit(0)

    if args.num_model_param == -1:
        print('Enter the number of model parameters, quitting...')
        sys.exit(0)

    # Modification
    list_file = os.path.join('submit', args.method, args.list_file, 'pred_npy_list.txt')
    submission_json = os.path.join('submit', args.method, args.list_file, args.submission_json + '.json')

    # create_json_from_npy_data(args.list_file, args.submission_json, args.num_model_param)
    create_json_from_npy_data(list_file, submission_json, args.num_model_param)

