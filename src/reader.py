# ---------------------------------------------------------
# Tensorflow Iris-GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import math
import tensorflow as tf


class Reader(object):
    def __init__(self, tfrecordsFile, decodeImgShape=(320, 400, 1), imgShape=(320, 200, 1), batchSize=1,
                 minQueueExamples=100, numThreads=8, name='DataReader'):
        self.tfrecordsFile = tfrecordsFile
        self.decodeImgShape = decodeImgShape
        self.imgShape = imgShape

        self.minQueueExamples = minQueueExamples
        self.batchSize = batchSize
        self.numThreads = numThreads
        self.reader = tf.TFRecordReader()
        self.name = name

        # For data augmentations
        self.resizeFactor = 1.1
        self.maxDelta = 255 * 0.1  # 'max_delta' must be in the interval [0, 0.5]
        self.rotateAngle = 10.
        self._graph()

    def _graph(self):
        with tf.compat.v1.variable_scope(self.name):
            filenameQueue = tf.train.string_input_producer([self.tfrecordsFile])

            _, serializedExample = self.reader.read(filenameQueue)
            features = tf.io.parse_single_example(serializedExample, features={
                'image/file_name': tf.io.FixedLenFeature([], tf.string),
                'image/user_id': tf.io.FixedLenFeature([], tf.string),
                'image/encoded_image': tf.io.FixedLenFeature([], tf.string)})

            imageBuffer = features['image/encoded_image']
            self.userIdBuffer = features['image/user_id']
            self.imageNameBuffer = features['image/file_name']
            image = tf.image.decode_jpeg(imageBuffer, channels=self.imgShape[2])

            # Resize to 2D
            image = tf.image.resize(image, size=(self.decodeImgShape[0], self.decodeImgShape[1]))

            # Split to two images
            self.imgOri, self.segImgOri = tf.split(image, num_or_size_splits=[self.imgShape[1], self.imgShape[1]],
                                                   axis=1)

    def shuffle_batch(self):
        img, segImg = self.preprocess(self.imgOri, self.segImgOri)

        return tf.train.shuffle_batch(tensors=[img, segImg],
                                      batch_size=self.batchSize,
                                      num_threads=self.numThreads,
                                      capacity=self.minQueueExamples + 3 * self.batchSize,
                                      min_after_dequeue=self.minQueueExamples)

    def batch(self, multi_test=False):
        if multi_test:
            img, segImg = self.multi_test_process(self.imgOri, self.segImgOri)
        else:
            img, segImg = self.imgOri, self.segImgOri

        return tf.train.batch(tensors=[img, segImg, self.imageNameBuffer, self.userIdBuffer],
                              batch_size=self.batchSize,
                              num_threads=self.numThreads,
                              capacity=self.minQueueExamples + 3 * self.batchSize,
                              allow_smaller_final_batch=True)

    def multi_test_process(self, imgOri, segImgOri):
        imgs, segImgs = list(), list()

        for degree in range(-10, 11, 2):
            img, segImg = self.fixed_rotation(imgOri, segImgOri, degree)
            imgs.append(img), segImgs.append(segImg)

        return imgs, segImgs


    def preprocess(self, imgOri, segImgOri):
        # Data augmentation
        imgTrans, segImgTrans = self.random_translation(imgOri, segImgOri)      # Random translation
        imgFlip, segImgFlip = self.random_flip(imgTrans, segImgTrans)           # Random left-right flip
        imgBrit, segImgBrit = self.random_brightness(imgFlip, segImgFlip)       # Random brightness
        imgRotate, segImgRotate = self.random_rotation(imgBrit, segImgBrit)     # Random rotation

        return imgRotate, segImgRotate

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

    def random_flip(self, img, segImg):
        # Step 1: Concat two images according to the depth axis
        combined = tf.concat(values=[img, segImg], axis=2)

        # Step 2: Random flip
        combined = tf.image.random_flip_left_right(image=combined)

        # Step 3: Split into two images
        img, segImg = tf.split(combined, num_or_size_splits=[self.imgShape[2], self.imgShape[2]], axis=2)

        return img, segImg

    def random_brightness(self, img, segImg):
        # Step 1: Random brightness
        img = tf.image.random_brightness(img, max_delta=self.maxDelta)

        # Step 2: Clip value in the range of v_min and v_max
        img = tf.clip_by_value(t=img, clip_value_min=0., clip_value_max=255.)

        return img, segImg

    def fixed_rotation(self, img, segImg, degree):
        # Step 1: Concat two images according to the depth axis
        combined = tf.concat(values=[img, segImg], axis=2)

        # Step 2: from degree to radian
        radian = degree * math.pi / 180.

        # Step 3: Rotate image
        combined = tf.contrib.image.rotate(images=combined, angles=radian, interpolation='NEAREST')

        # Step 4: Split into two images
        img, segImg = tf.split(combined, num_or_size_splits=[self.imgShape[2], self.imgShape[2]], axis=2)

        return img, segImg

    def random_rotation(self, img, segImg):
        # Step 1: Concat two images according to the depth axis
        combined = tf.concat(values=[img, segImg], axis=2)

        # Step 2: Select a random angle
        radian_min = -self.rotateAngle * math.pi / 180.
        radian_max = self.rotateAngle * math.pi / 180.
        random_angle = tf.random.uniform(shape=[1], minval=radian_min, maxval=radian_max)

        # Step 3: Rotate image
        combined = tf.contrib.image.rotate(images=combined, angles=random_angle, interpolation='NEAREST')

        # Step 4: Split into two images
        img, segImg = tf.split(combined, num_or_size_splits=[self.imgShape[2], self.imgShape[2]], axis=2)

        return img, segImg

    def test_random_translation(self, numImgs):
        imgs, segImgs = list(), list()
        for i in range(numImgs):
            img, segImg = self.random_translation(self.imgOri, self.segImgOri)
            imgs.append(img), segImgs.append(segImg)

        return tf.train.shuffle_batch(tensors=[imgs, segImgs, self.imgOri, self.segImgOri],
                                      batch_size=self.batchSize,
                                      num_threads=self.numThreads,
                                      capacity=self.minQueueExamples + 3 * self.batchSize,
                                      min_after_dequeue=self.minQueueExamples)

    def test_random_flip(self, numImgs):
        imgs, segImgs = list(), list()
        for i in range(numImgs):
            img, segImg = self.random_flip(self.imgOri, self.segImgOri)
            imgs.append(img), segImgs.append(segImg)

        return tf.train.shuffle_batch(tensors=[imgs, segImgs, self.imgOri, self.segImgOri],
                                      batch_size=self.batchSize,
                                      num_threads=self.numThreads,
                                      capacity=self.minQueueExamples + 3 * self.batchSize,
                                      min_after_dequeue=self.minQueueExamples)

    def test_random_brightness(self, numImgs):
        imgs, segImgs = list(), list()
        for i in range(numImgs):
            img, segImg = self.random_brightness(self.imgOri, self.segImgOri)
            imgs.append(img), segImgs.append(segImg)

        return tf.train.shuffle_batch(tensors=[imgs, segImgs, self.imgOri, self.segImgOri],
                                      batch_size=self.batchSize,
                                      num_threads=self.numThreads,
                                      capacity=self.minQueueExamples + 3 * self.batchSize,
                                      min_after_dequeue=self.minQueueExamples)

    def test_random_rotation(self, numImgs):
        imgs, segImgs = list(), list()
        for i in range(numImgs):
            img, segImg = self.random_rotation(self.imgOri, self.segImgOri)
            imgs.append(img), segImgs.append(segImg)

        return tf.train.shuffle_batch(tensors=[imgs, segImgs, self.imgOri, self.segImgOri],
                                      batch_size=self.batchSize,
                                      num_threads=self.numThreads,
                                      capacity=self.minQueueExamples + 3 * self.batchSize,
                                      min_after_dequeue=self.minQueueExamples)

    def test_multi_test(self):
        imgs, segImgs = self.multi_test_process(self.imgOri, self.segImgOri)

        return tf.train.shuffle_batch(tensors=[imgs, segImgs],
                                      batch_size=self.batchSize,
                                      num_threads=self.numThreads,
                                      capacity=self.minQueueExamples + 3 * self.batchSize,
                                      min_after_dequeue=self.minQueueExamples)
