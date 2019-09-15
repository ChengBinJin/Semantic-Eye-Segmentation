import os
import math
import cv2
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_utils as tf_utils
from utils import JsonData, all_files_under

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path_list', dest='data_path_list', type=str, default='input_folder_list.txt',
                    help='input data folder addresses are listed in the txt file')
parser.add_argument('--is_debug', dest='is_debug', default=True, action='store_true',
                    help='debug for saving labels as the color images')
args = parser.parse_args()


class UNet(object):
    def __init__(self, input_shape=(640, 400, 1), num_classes=4, name='UNet'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = name
        self._ops = list()
        self.max_degree = 10
        self.interval = 2
        # multi_test: from -10 degree to 10 degree with inteval and lef-right flipping
        self.num_try = 2*len(range(-self.max_degree, self.max_degree+1, self.interval))
        self.conv_dims = [32, 32, 32, 32, 32, 32, 32, 32, 64, 64,
                         32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16]
        self._build_graph()  # main graph

    def _build_graph(self):
        # Input placeholders
        self.input_img_tfph = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.input_shape], name='input_img_tfph')
        self.input_img = self.multi_test_graph(self.input_img_tfph)
        print('self.input_img shape: {}'.format(self.input_img.get_shape().as_list()))

        # Network forward
        self.preds = self.forward_network(img=self.normalize(self.input_img))
        # self.pred_cls = tf.math.argmax(self.pred, axis=-1)

        # Step 1: original rotated images
        preds_s1 = self.preds

        # Step 2: inverse-rotated images
        preds_s2 = self.inverse_rotation(preds_s1)

        # Step 3: combine all results to estimate the final result
        sum_all = tf.math.reduce_sum(preds_s2, axis=0)      # [N, H, W, num_classes] -> [H, W, num_classes]
        self.pred_cls = tf.math.argmax(sum_all, axis=-1)    # [H, W, num_classes] -> [H, W]

    def inverse_rotation(self, preds):
        preds_inv = list()

        num_imgs, h, w, c = preds.get_shape().as_list()
        n_flip = 2  # num of ori-img and flip-img
        n_rotate = len(range(-self.max_degree, self.max_degree+1, self.interval))

        for i in range(n_flip):     # ori-img and flipped img
            for idx, degree in enumerate(range(self.max_degree, -self.max_degree-1, -self.interval)):
                # Extract specific tensor
                pred = tf.slice(preds, begin=[idx + i * n_rotate, 0, 0, 0], size=[1, h, w, c])
                # From degree to radian
                radian = degree * math.pi / 180.
                # Rotate pred
                pred_inv = tf.contrib.image.rotate(images=pred, angles=radian, interpolation='NEAREST')

                if i == 1:
                    # Flipping flipped images
                    pred_inv = tf.image.flip_left_right(pred_inv)

                preds_inv.append(pred_inv)

        output = tf.concat(preds_inv, axis=0)
        return output

    def multi_test_graph(self, img_ori):
        imgs = list()
        for img in [img_ori, tf.image.flip_left_right(img_ori)]:
            for degree in range(-self.max_degree, self.max_degree+1, self.interval):
                pre_img = self.fixed_rotation(img, degree)
                imgs.append(pre_img)

        # List to tensor and reahpe to the required input
        output = tf.reshape(tf.convert_to_tensor(imgs), (self.num_try, *self.input_shape))
        return output

    @staticmethod
    def fixed_rotation(img, degree):
        # Step 1: from degree to radian
        radian = degree * math.pi / 180.
        # Step 2: Rotate image
        rotated_img = tf.contrib.image.rotate(images=img, angles=radian, interpolation='BILINEAR')
        return rotated_img

    @staticmethod
    def normalize(data):
        return data / 127.5 - 1.0

    def forward_network(self, img, padding='SAME', reuse=False):
        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            # Stage 0
            s0_conv1 = tf_utils.conv2d(x=img, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s0_conv1')
            s0_conv1 = tf_utils.relu(s0_conv1, name='relu_s0_conv1')

            s0_conv2 = tf_utils.conv2d(x=s0_conv1, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s0_conv2')
            s0_conv2 = tf_utils.norm(s0_conv2, name='s0_norm1', _type='batch', _ops=self._ops, is_train=False)
            s0_conv2 = tf_utils.relu(s0_conv2, name='relu_s0_conv2')

            # Stage 1
            s1_maxpool = tf_utils.max_pool(x=s0_conv2, name='s1_maxpool2d')

            s1_conv1 = tf_utils.conv2d(x=s1_maxpool, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s1_conv1')
            s1_conv1 = tf_utils.norm(s1_conv1, name='s1_norm0', _type='batch', _ops=self._ops, is_train=False)
            s1_conv1 = tf_utils.relu(s1_conv1, name='relu_s1_conv1')

            s1_conv2 = tf_utils.conv2d(x=s1_conv1, output_dim=self.conv_dims[1], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s1_conv2')
            s1_conv2 = tf_utils.norm(s1_conv2, name='s1_norm1', _type='batch', _ops=self._ops, is_train=False)
            s1_conv2 = tf_utils.relu(s1_conv2, name='relu_s1_conv2')

            # Stage 2
            s2_maxpool = tf_utils.max_pool(x=s1_conv2, name='s2_maxpool2d')
            s2_conv1 = tf_utils.conv2d(x=s2_maxpool, output_dim=self.conv_dims[2], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s2_conv1')
            s2_conv1 = tf_utils.norm(s2_conv1, name='s2_norm0', _type='batch', _ops=self._ops, is_train=False)
            s2_conv1 = tf_utils.relu(s2_conv1, name='relu_s2_conv1')

            s2_conv2 = tf_utils.conv2d(x=s2_conv1, output_dim=self.conv_dims[3], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s2_conv2')
            s2_conv2 = tf_utils.norm(s2_conv2, name='s2_norm1', _type='batch', _ops=self._ops, is_train=False)
            s2_conv2 = tf_utils.relu(s2_conv2, name='relu_s2_conv2')

            # Stage 3
            s3_maxpool = tf_utils.max_pool(x=s2_conv2, name='s3_maxpool2d')
            s3_conv1 = tf_utils.conv2d(x=s3_maxpool, output_dim=self.conv_dims[4], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s3_conv1')
            s3_conv1 = tf_utils.norm(s3_conv1, name='s3_norm0', _type='batch', _ops=self._ops, is_train=False)
            s3_conv1 = tf_utils.relu(s3_conv1, name='relu_s3_conv1')

            s3_conv2 = tf_utils.conv2d(x=s3_conv1, output_dim=self.conv_dims[5], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s3_conv2')
            s3_conv2 = tf_utils.norm(s3_conv2, name='s3_norm1', _type='batch', _ops=self._ops, is_train=False)
            s3_conv2 = tf_utils.relu(s3_conv2, name='relu_s3_conv2')

            # Stage 4
            s4_maxpool = tf_utils.max_pool(x=s3_conv2, name='s4_maxpool2d')
            s4_conv1 = tf_utils.conv2d(x=s4_maxpool, output_dim=self.conv_dims[6], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s4_conv1')
            s4_conv1 = tf_utils.norm(s4_conv1, name='s4_norm0', _type='batch', _ops=self._ops, is_train=False)
            s4_conv1 = tf_utils.relu(s4_conv1, name='relu_s4_conv1')

            s4_conv2 = tf_utils.conv2d(x=s4_conv1, output_dim=self.conv_dims[7], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s4_conv2')
            s4_conv2 = tf_utils.norm(s4_conv2, name='s4_norm1', _type='batch', _ops=self._ops, is_train=False)
            s4_conv2 = tf_utils.relu(s4_conv2, name='relu_s4_conv2')
            s4_conv2_drop = tf_utils.dropout(x=s4_conv2, keep_prob=0., name='s4_dropout')

            # Stage 5
            s5_maxpool = tf_utils.max_pool(x=s4_conv2_drop, name='s5_maxpool2d')
            s5_conv1 = tf_utils.conv2d(x=s5_maxpool, output_dim=self.conv_dims[8], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s5_conv1')
            s5_conv1 = tf_utils.norm(s5_conv1, name='s5_norm0', _type='batch', _ops=self._ops, is_train=False)
            s5_conv1 = tf_utils.relu(s5_conv1, name='relu_s5_conv1')

            s5_conv2 = tf_utils.conv2d(x=s5_conv1, output_dim=self.conv_dims[9], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s5_conv2')
            s5_conv2 = tf_utils.norm(s5_conv2, name='s5_norm1', _type='batch', _ops=self._ops, is_train=False)
            s5_conv2 = tf_utils.relu(s5_conv2, name='relu_s5_conv2')
            s5_conv2_drop = tf_utils.dropout(x=s5_conv2, keep_prob=0., name='s5_dropout')

            # Stage 6
            s6_deconv1 = tf_utils.deconv2d(x=s5_conv2_drop, output_dim=self.conv_dims[10], k_h=2, k_w=2,
                                           initializer='He', name='s6_deconv1')
            s6_deconv1 = tf_utils.norm(s6_deconv1, name='s6_norm0', _type='batch', _ops=self._ops, is_train=False)
            s6_deconv1 = tf_utils.relu(s6_deconv1, name='relu_s6_deconv1')
            # Cropping
            w1 = s4_conv2_drop.get_shape().as_list()[2]
            w2 = s6_deconv1.get_shape().as_list()[2] - s4_conv2_drop.get_shape().as_list()[2]
            s6_deconv1_split, _ = tf.split(s6_deconv1, num_or_size_splits=[w1, w2], axis=2, name='axis2_split')
            tf_utils.print_activations(s6_deconv1_split)
            # Concat
            s6_concat = tf_utils.concat(values=[s6_deconv1_split, s4_conv2_drop], axis=3, name='s6_axis3_concat')

            s6_conv2 = tf_utils.conv2d(x=s6_concat, output_dim=self.conv_dims[11], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s6_conv2')
            s6_conv2 = tf_utils.norm(s6_conv2, name='s6_norm1', _type='batch', _ops=self._ops, is_train=False)
            s6_conv2 = tf_utils.relu(s6_conv2, name='relu_s6_conv2')

            s6_conv3 = tf_utils.conv2d(x=s6_conv2, output_dim=self.conv_dims[12], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s6_conv3')
            s6_conv3 = tf_utils.norm(s6_conv3, name='s6_norm2', _type='batch', _ops=self._ops, is_train=False)
            s6_conv3 = tf_utils.relu(s6_conv3, name='relu_s6_conv3')

            # Stage 7
            s7_deconv1 = tf_utils.deconv2d(x=s6_conv3, output_dim=self.conv_dims[13], k_h=2, k_w=2, initializer='He',
                                           name='s7_deconv1')
            s7_deconv1 = tf_utils.norm(s7_deconv1, name='s7_norm0', _type='batch', _ops=self._ops, is_train=False)
            s7_deconv1 = tf_utils.relu(s7_deconv1, name='relu_s7_deconv1')
            # Concat
            s7_concat = tf_utils.concat(values=[s7_deconv1, s3_conv2], axis=3, name='s7_axis3_concat')

            s7_conv2 = tf_utils.conv2d(x=s7_concat, output_dim=self.conv_dims[14], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s7_conv2')
            s7_conv2 = tf_utils.norm(s7_conv2, name='s7_norm1', _type='batch', _ops=self._ops, is_train=False)
            s7_conv2 = tf_utils.relu(s7_conv2, name='relu_s7_conv2')

            s7_conv3 = tf_utils.conv2d(x=s7_conv2, output_dim=self.conv_dims[15], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s7_conv3')
            s7_conv3 = tf_utils.norm(s7_conv3, name='s7_norm2', _type='batch', _ops=self._ops, is_train=False)
            s7_conv3 = tf_utils.relu(s7_conv3, name='relu_s7_conv3')

            # Stage 8
            s8_deconv1 = tf_utils.deconv2d(x=s7_conv3, output_dim=self.conv_dims[16], k_h=2, k_w=2, initializer='He',
                                           name='s8_deconv1')
            s8_deconv1 = tf_utils.norm(s8_deconv1, name='s8_norm0', _type='batch', _ops=self._ops, is_train=False)
            s8_deconv1 = tf_utils.relu(s8_deconv1, name='relu_s8_deconv1')
            # Concat
            s8_concat = tf_utils.concat(values=[s8_deconv1,s2_conv2], axis=3, name='s8_axis3_concat')

            s8_conv2 = tf_utils.conv2d(x=s8_concat, output_dim=self.conv_dims[17], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s8_conv2')
            s8_conv2 = tf_utils.norm(s8_conv2, name='s8_norm1', _type='batch', _ops=self._ops,
                                         is_train=False)
            s8_conv2 = tf_utils.relu(s8_conv2, name='relu_s8_conv2')

            s8_conv3 = tf_utils.conv2d(x=s8_conv2, output_dim=self.conv_dims[18], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s8_conv3')
            s8_conv3 = tf_utils.norm(s8_conv3, name='s8_norm2', _type='batch', _ops=self._ops, is_train=False)
            s8_conv3 = tf_utils.relu(s8_conv3, name='relu_conv3')

            # Stage 9
            s9_deconv1 = tf_utils.deconv2d(x=s8_conv3, output_dim=self.conv_dims[19], k_h=2, k_w=2, initializer='He',
                                           name='s9_deconv1')
            s9_deconv1 = tf_utils.norm(s9_deconv1, name='s9_norm0', _type='batch', _ops=self._ops, is_train=False)
            s9_deconv1 = tf_utils.relu(s9_deconv1, name='relu_s9_deconv1')
            # Concat
            s9_concat = tf_utils.concat(values=[s9_deconv1, s1_conv2], axis=3, name='s9_axis3_concat')

            s9_conv2 = tf_utils.conv2d(x=s9_concat, output_dim=self.conv_dims[20], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s9_conv2')
            s9_conv2 = tf_utils.norm(s9_conv2, name='s9_norm1', _type='batch', _ops=self._ops, is_train=False)
            s9_conv2 = tf_utils.relu(s9_conv2, name='relu_s9_conv2')

            s9_conv3 = tf_utils.conv2d(x=s9_conv2, output_dim=self.conv_dims[21], k_h=3, k_w=3, d_h=1, d_w=1,
                                       padding=padding, initializer='He', name='s9_conv3')
            s9_conv3 = tf_utils.norm(s9_conv3, name='s9_norm2', _type='batch', _ops=self._ops, is_train=False)
            s9_conv3 = tf_utils.relu(s9_conv3, name='relu_s9_conv3')

            s10_deconv1 = tf_utils.deconv2d(x=s9_conv3, output_dim=self.conv_dims[-1], k_h=2, k_w=2,
                                            initializer='He', name='s10_deconv1')
            s10_deconv1 = tf_utils.norm(s10_deconv1, name='s10_norm0', _type='batch', _ops=self._ops, is_train=False)
            s10_deconv1 = tf_utils.relu(s10_deconv1, name='relu_s10_deconv1')
            # Concat
            s10_concat = tf_utils.concat(values=[s10_deconv1, s0_conv2], axis=3, name='s10_axis3_concat')

            s10_conv2 = tf_utils.conv2d(s10_concat, output_dim=self.conv_dims[-1], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s10_conv2')
            s10_conv2 = tf_utils.norm(s10_conv2, name='s10_norm1', _type='batch', _ops=self._ops, is_train=False)
            s10_conv2 = tf_utils.relu(s10_conv2, name='relu_s10_conv2')

            s10_conv3 = tf_utils.conv2d(x=s10_conv2, output_dim=self.conv_dims[-1], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s10_conv3')
            s10_conv3 = tf_utils.norm(s10_conv3, name='s10_norm2', _type='batch', _ops=self._ops, is_train=False)
            s10_conv3 = tf_utils.relu(s10_conv3, name='relu_s10_conv3')

            output = tf_utils.conv2d(s10_conv3, output_dim=self.num_classes, k_h=1, k_w=1, d_h=1, d_w=1,
                                     padding=padding, initializer='He', name='output')

            return output


class Solver(object):
    def __init__(self, model, method='U-Net-light-v4_2', model_name='20190816-230440'):
        self.model = model
        self.model_dir = os.path.join('../model', method, model_name)
        self._init_session()
        self._init_variables()

    def _init_session(self):
        self.sess = tf.compat.v1.Session()

    def _init_variables(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def test(self, img):
        img = np.expand_dims(np.expand_dims(img, axis=0), axis=3)
        pred = self.sess.run(self.model.pred_cls, feed_dict={self.model.input_img_tfph: img})
        return np.squeeze(pred)

    def load_model(self):
        # Initialize saver
        saver = tf.compat.v1.train.Saver(max_to_keep=1)

        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(self.model_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            return True, iter_time
        else:
            return False, None


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


def save_npy(data, save_dir, file_name):
    # Convert data type from int32 to uint8
    data = data.astype(np.uint8)
    # Save data in npy format by requirement
    np.save(os.path.join(save_dir, file_name), data)


def read_data(data_path_list):
    file = open(data_path_list, 'r')
    paths = file.readlines()
    json_obj = JsonData()

    overall_paths = list()
    overall_user_id = list()
    for i, path in enumerate(paths):
        path = path.strip()
        stage = os.path.dirname(path).split('/')[-2]

        img_paths = all_files_under(folder=path, subfolder=None, endswith='.png')
        overall_paths.extend(img_paths)
        print('{}: {} - num. of images: {}'.format(i, path, len(img_paths)))

        for j, img_path in enumerate(img_paths):
            # TODO: key shoulde be adaptive
            _, user_id = json_obj.find_id(target=os.path.basename(img_path),
                                          data_set=stage,
                                          key='generative_images')

            overall_user_id.extend([user_id])

            if j % 500 == 0:
                print('Reading {}/{}...'.format(j, len(img_paths)))

    return overall_paths, overall_user_id


def save_img(img, pred, save_dir, img_name, user_id):
    img_name = img_name.replace('.png', '_'+user_id+'.png')

    h, w = img.shape
    canvas = np.zeros((h, 2*w, 3), np.uint8)
    canvas[:, :w, :] = np.dstack((img, img, img))
    canvas[:, w:, :] = convert_color_label(pred)
    cv2.imwrite(os.path.join(save_dir, img_name), canvas)


def main(data_path_list, is_debug=True):
    # Read the all images in the defined folder
    img_paths, user_ids = read_data(data_path_list)

    # Initialize the network and solver
    model = UNet()
    solver = Solver(model)
    # Read a checkpoint to restore the model
    flag, iter_time = solver.load_model()
    if flag is True:
        print(' [!] Load Success! Iter: {}'.format(iter_time))
    else:
        exit(' [!] Failed to restore model'.format(solver.model_dir))

    for i, (img_path, user_id) in enumerate(zip(img_paths, user_ids)):
        img_name = os.path.basename(img_path)
        save_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if i % 20 == 0:
            print('Processing {} / {}...'.format(i+1, len(img_paths)))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        pred = solver.test(img)
        save_npy(pred, save_dir, img_name)

        if is_debug:
            save_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'paired')
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            save_img(img, pred, save_dir, img_name, user_id)


if __name__ == '__main__':
    main(data_path_list=args.data_path_list, is_debug=args.is_debug)


