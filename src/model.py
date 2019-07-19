# --------------------------------------------------------------------------
# Tensorflow Implementation of OpenEDS Semantic Segmentation Challenge
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# -------------------------------------------------------------------------

class UNet(object):
    def __init__(self, imageShape=(640, 800, 1), outputShape=(640, 400, 1), resizeFactor=0.5, dataPath=[None, None],
                 batchSize=1, lr=1e-3, weight_decay=1e-4, total_iters=2e5, isTrain=True, logDir=None, name='UNet'):
        print("Hello UNet!")
