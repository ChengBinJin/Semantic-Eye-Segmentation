# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import math
import tensorflow as tf


class Reader(object):
    def __init__(self, tfrecordsFile, decodeImgShape=(320, 400, 1), imgShape=(320, 200, 1), batchSize=1, isTrain=True,
                 minQueueExamples=100, numThreads=8, name='DataReader'):
        self.tfrecordsFile = tfrecordsFile
        self.decodeImgShape = decodeImgShape
        self.imgShape = imgShape

        self.minQueueExamples = minQueueExamples
        self.batchSize = batchSize
        self.numThreads = numThreads
        self.reader = tf.TFRecordReader()
        self.isTrain = isTrain
        self.name = name

        # For data augmentations
        self.resizeFactor = 1.1
        self.rotateAngle = 5.
        self._graph()

    def _graph(self):
        with tf.name_scope(self.name):
            filenameQueue = tf.train.string_input_producer([self.tfrecordsFile])

            _, serializedExample = self.reader.read(filenameQueue)
            features = tf.io.parse_single_example(serializedExample, features={
                'image/file_name': tf.io.FixedLenFeature([], tf.string),
                'image/user_id': tf.io.FixedLenFeature([], tf.string),
                'image/encoded_image': tf.io.FixedLenFeature([], tf.string)})

            imageBuffer = features['image/encoded_image']
            # userIdBuffer = features['image/user_id']
            # imageNameBuffer = features['image/file_name']
            self.image = tf.image.decode_jpeg(imageBuffer, channels=self.imgShape[2])

    def shuffle_batch(self):
        # img_ori, img_trans, img_flip, img_rotate = self.preprocess(image, is_train=self.is_train)
        img, segImg, imgOri, segImgOri = self.preprocess(self.image, isTrain=self.isTrain)

        return tf.train.shuffle_batch(tensors=[img, segImg, imgOri, segImgOri],
                                      batch_size=self.batchSize,
                                      num_threads=self.numThreads,
                                      capacity=self.minQueueExamples + 3 * self.batchSize,
                                      min_after_dequeue=self.minQueueExamples)

    def preprocess(self, image, isTrain=True):
        # Resize to 2D
        image = tf.image.resize(image, size=(self.decodeImgShape[0], self.decodeImgShape[1]))
        # Split to two images
        imgOri, segImgOri = tf.split(image, num_or_size_splits=[self.imgShape[1], self.imgShape[1]], axis=1)

        # Data augmentation
        if isTrain:
            imgTrans, segImgTrans = self.random_translation(imgOri, segImgOri)     # Random translation
            # img_flip = self.RandomFlip(img_trans)           # Random left-right flip
            # img_rotate = self.RandomRotation(img_flip)      # Random rotation

        else:
            imgTrans = imgOri
            segImgTrans = segImgOri

        return imgTrans, segImgTrans, imgOri, segImgOri

    def random_translation(self, img, segImg):
        # Step 1: Resized to the bigger image
        img = tf.image.resize(images=img,
                              size=(int(self.resizeFactor * self.imgShape[0]),
                                    int(self.resizeFactor * self.imgShape[1])),
                              method=tf.image.ResizeMethod.BICUBIC)
        segImg = tf.image.resize(images=segImg,
                                 size=(int(self.resizeFactor * self.imgShape[0]),
                                       int(self.resizeFactor * self.imgShape[1])),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Step 2: Concat two images according to the depth axis
        combined = tf.concat(values=[img, segImg], axis=2)
        depth = combined.get_shape().as_list()[-1]

        # Step 3: Random crop
        combined = tf.image.random_crop(value=combined, size=[*self.imgShape[0:2], depth])

        # Step 4: Clip value in the range of v_min and v_max
        combined = tf.clip_by_value(t=combined, clip_value_min=0., clip_value_max=255.)

        # Step 5: Split into two images
        img, segImg = tf.split(combined, num_or_size_splits=[self.imgShape[2], self.imgShape[2]], axis=2)

        return img, segImg

    @staticmethod
    def RandomFlip(img_ori, is_random=True):
        if is_random:
            img = tf.image.random_flip_left_right(image=img_ori)
        else:
            img = tf.image.flip_left_right(img_ori)

        return img

    def RandomRotation(self, img_ori):
        radian_min = -self.rotateAngle * math.pi / 180.
        radian_max = self.rotateAngle * math.pi / 180.
        random_angle = tf.random.uniform(shape=[1], minval=radian_min, maxval=radian_max)
        img = tf.contrib.image.rotate(images=img_ori, angles=random_angle, interpolation='BILINEAR')

        return img

    def test_random_translation(self, numImgs):
        # Resize to 2D
        image = tf.image.resize(self.image, size=(self.decodeImgShape[0], self.decodeImgShape[1]))
        # Split to two images
        imgOri, segImgOri = tf.split(image, num_or_size_splits=[self.imgShape[1], self.imgShape[1]], axis=1)

        imgs, segImgs = list(), list()
        for i in range(numImgs):
            img, segImg = self.random_translation(imgOri, segImgOri)
            imgs.append(img), segImgs.append(segImg)

        return tf.train.shuffle_batch(tensors=[imgs, segImgs, imgOri, segImgOri],
                                      batch_size=self.batchSize,
                                      num_threads=self.numThreads,
                                      capacity=self.minQueueExamples + 3 * self.batchSize,
                                      min_after_dequeue=self.minQueueExamples)
