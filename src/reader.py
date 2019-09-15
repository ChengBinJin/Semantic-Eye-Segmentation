# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------
import math
import tensorflow as tf

class ReaderIdentity(object):
    def __init__(self, tfrecords_file, decode_img_shape=(320, 200, 3), batch_size=1, min_queue_examples=1000, num_threads=8,
                 name='Identity'):
        self.tfrecords_file = tfrecords_file
        self.decode_img_shape = decode_img_shape

        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.name = name

        # Note: We don't data augmentation for the iris identification
        self._graph()

    def _graph(self):
        with tf.compat.v1.variable_scope(self.name):
            filenameQueue = tf.train.string_input_producer([self.tfrecords_file])

            _, serializedExample = self.reader.read(filenameQueue)
            features = tf.io.parse_single_example(serializedExample, features={
                'image/img_name': tf.io.FixedLenFeature([], tf.string),
                'image/cls_number': tf.io.FixedLenFeature([], tf.string),
                'image/encoded_image': tf.io.FixedLenFeature([], tf.string)})

            # Buffers
            img_buffer = features['image/encoded_image']
            self.cls_number_buffer = features['image/cls_number']
            self.img_name_buffer = features['image/img_name']

            # Resize to 2D
            img = tf.image.decode_jpeg(img_buffer, channels=self.decode_img_shape[2])
            img = tf.image.resize(img, size=(self.decode_img_shape[0], self.decode_img_shape[1]))
            # Extract green channel img
            _, self.img, _ = tf.split(img, num_or_size_splits=[1, 1, 1], axis=-1)

    def shuffle_batch(self):
        return tf.train.shuffle_batch(tensors=[self.img, self.cls_number_buffer],
                                      batch_size=self.batch_size,
                                      num_threads=self.num_threads,
                                      capacity=self.min_queue_examples + self.num_threads * self.batch_size,
                                      min_after_dequeue=self.min_queue_examples)

    def batch(self, multi_test=False):
        return tf.train.batch(tensors=[self.img, self.cls_number_buffer],
                              batch_size=self.batch_size,
                              num_threads=self.num_threads,
                              capacity=self.min_queue_examples + self.num_threads * self.batch_size,
                              allow_smaller_final_batch=True)


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
            # image = tf.image.resize(image, size=(self.decodeImgShape[0], self.decodeImgShape[1]))
            image = tf.cast(tf.image.resize(image, size=(self.decodeImgShape[0], self.decodeImgShape[1]),
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), dtype=tf.float32)

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

    def batch(self, multi_test=False, use_advanced=False, interval=1):
        if multi_test:
            if use_advanced:
                img, segImg = self.multi_test_process_advanced(self.imgOri, self.segImgOri, interval=interval)
            else:
                img, segImg = self.multi_test_process(self.imgOri, self.segImgOri)
        else:
            img, segImg = self.imgOri, self.segImgOri

        return tf.train.batch(tensors=[img, segImg, self.imageNameBuffer, self.userIdBuffer],
                              batch_size=self.batchSize,
                              num_threads=self.numThreads,
                              capacity=self.minQueueExamples + 3 * self.batchSize,
                              allow_smaller_final_batch=True)

    def multi_test_process_advanced(self, imgOri, segImgOri, interval=1):
        hMargin = int(self.resizeFactor * self.imgShape[0]) - self.imgShape[0]
        wMargin = int(self.resizeFactor * self.imgShape[1]) - self.imgShape[1]

        imgs, segImgs = list(), list()
        flipImgs, flipSegImgs = list(), list()

        # Step1: Cropping & flipping
        # Original image
        imgs.append(imgOri); flipImgs.append(tf.image.flip_left_right(imgOri))
        segImgs.append(segImgOri); flipSegImgs.append(tf.image.flip_left_right(segImgOri))

        # Resized to the bigger image
        img = tf.image.resize(images=imgOri,
                              size=(int(self.resizeFactor * self.imgShape[0]),
                                    int(self.resizeFactor * self.imgShape[1])),
                              method=tf.image.ResizeMethod.BICUBIC)
        img = tf.clip_by_value(t=img, clip_value_min=0., clip_value_max=255.)

        segImg = tf.image.resize(images=segImgOri,
                                 size=(int(self.resizeFactor * self.imgShape[0]),
                                       int(self.resizeFactor * self.imgShape[1])),
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Top-left image
        tpImg = tf.slice(img, begin=[0, 0, 0], size=[*self.imgShape])
        imgs.append(tpImg); flipImgs.append(tf.image.flip_left_right(tpImg))

        tpSegImg = tf.slice(segImg, begin=[0, 0, 0], size=[*self.imgShape])
        segImgs.append(tpSegImg); flipSegImgs.append(tf.image.flip_left_right(tpSegImg))

        # Top-right image
        trImg = tf.slice(img, begin=[0, wMargin, 0], size=[*self.imgShape])
        imgs.append(trImg); flipImgs.append(tf.image.flip_left_right(trImg))

        trSegImg = tf.slice(segImg, begin=[0, wMargin, 0], size=[*self.imgShape])
        segImgs.append(trSegImg); flipSegImgs.append(tf.image.flip_left_right(trSegImg))

        # Center image
        cImg = tf.slice(img, begin=[int(hMargin * 0.5), int(wMargin * 0.5), 0], size=[*self.imgShape])
        imgs.append(cImg); flipImgs.append(tf.image.flip_left_right(cImg))

        cSegImg = tf.slice(segImg, begin=[int(hMargin * 0.5), int(wMargin * 0.5), 0], size=[*self.imgShape])
        segImgs.append(cSegImg); flipSegImgs.append(tf.image.flip_left_right(cSegImg))

        # Bottom-left image
        blImg = tf.slice(img, begin=[hMargin, 0, 0], size=[*self.imgShape])
        imgs.append(blImg); flipImgs.append(tf.image.flip_left_right(blImg))

        blSegImg = tf.slice(segImg, begin=[hMargin, 0, 0], size=[*self.imgShape])
        segImgs.append(blSegImg); flipSegImgs.append(tf.image.flip_left_right(blSegImg))

        # Bottom-right image
        brImg = tf.slice(img, begin=[hMargin, wMargin, 0], size=[*self.imgShape])
        imgs.append(brImg); flipImgs.append(tf.image.flip_left_right(brImg))

        brSegImg = tf.slice(segImg, begin=[hMargin, wMargin, 0], size=[*self.imgShape])
        segImgs.append(brSegImg); flipSegImgs.append(tf.image.flip_left_right(brSegImg))

        imgs_, segImgs_ = list(), list()
        for imgOri_, segImgOri_ in zip(imgs + flipImgs, segImgs + flipSegImgs):
            for degree in range(-int(self.rotateAngle), int(self.rotateAngle + 1), interval):
                img, segImg = self.fixed_rotation(imgOri_, segImgOri_, degree)
                imgs_.append(img), segImgs_.append(segImg)

        return imgs_, segImgs_

    def multi_test_process(self, imgOri, segImgOri):
        imgs, segImgs = list(), list()

        for (imgOri_, segImgOri_) in [(imgOri, segImgOri),
                                     (tf.image.flip_left_right(imgOri), tf.image.flip_left_right(segImgOri))]:
            for degree in range(-10, 11, 2):
                img, segImg = self.fixed_rotation(imgOri_, segImgOri_, degree)
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

    def test_advanced_multi_test(self):
        imgs, segImgs = self.multi_test_process_advanced(self.imgOri, self.segImgOri)

        return tf.train.shuffle_batch(tensors=[imgs, segImgs],
                                      batch_size=self.batchSize,
                                      num_threads=self.numThreads,
                                      capacity=self.minQueueExamples + 3 * self.batchSize,
                                      min_after_dequeue=self.minQueueExamples)
