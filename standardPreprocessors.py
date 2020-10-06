import numpy as np
import tensorflow as tf
import cv2


import BasicInputPreprocessing



class RandomBrightness(BasicInputPreprocessing.BasicInputPreprocessing):
	def __init__(self,brightnessRange=0.3, parameters=None):
		super().__init__(str(type(self).__name__))
		if (parameters is None):
			self.parameters['brightnessRange'] = brightnessRange
		else:
			self.parameters = parameters



	def preprocessImage(self, img):
		return tf.image.random_brightness(img, self.parameters['brightnessRange'])


class RandomContrast(BasicInputPreprocessing.BasicInputPreprocessing):
	def __init__(self,contrastRange=0.5, parameters=None):
		super().__init__(str(type(self).__name__))
		if (parameters is None):
			self.parameters['contrastRange'] = contrastRange
		else:
			self.parameters = parameters



	def preprocessImage(self, img):
		return tf.image.random_contrast(img, 1 - self.parameters['contrastRange'], 1 + self.parameters['contrastRange'])


class tfPreprocessing(BasicInputPreprocessing.BasicInputPreprocessing):
	def __init__(self,preproctype='standard', parameters=None):
		super().__init__(str(type(self).__name__))
		if (parameters is None):
			self.parameters['preproctype'] = preproctype
		else:
			self.parameters = parameters


	def preprocessImage(self, img):
		return tf.image.per_image_standardization(img)

