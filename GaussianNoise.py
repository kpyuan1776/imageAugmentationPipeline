import numpy as np
import tensorflow as tf
import cv2


import BasicImagePairAugmentation


class GaussianNoise(BasicImagePairAugmentation.BasicImagePairAugmentation):
    """ adds gaussian white noise to image only (not to mask)
    """

    def __init__(self,noise_std=0.1,seed=None, parameters=None):
        super().__init__(str(type(self).__name__))
        self.seed=seed
        if (parameters is None):
			self.parameters['noise_std'] = noise_std
		else:
			self.parameters = parameters


    def modifyImagePair(self, inputImg, outputImg):
        noisyImg = tf.add(tf.cast(inputImg, tf.float32), tf.random.normal(shape=tf.shape(inputImg), mean=0.0, 
                stddev=self.parameters['noise_std'], dtype=tf.float32, seed=self.seed))
        return noisyImg,outputImg
