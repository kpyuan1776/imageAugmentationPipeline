
import numpy as np
import tensorflow as tf
import cv2

import BasicImageCropper


class RandomPosZoomCropper(BasicImageCropper.BasicImageCropper):
	""" implements cropImagePair() and returns a cropped image and mask tuple of fixed height,width 
		defined by patchSize. The crop has a random zoom level as well to make the training more robust against size scales.
		ARGS: 
			zoomRange: width of interval for allowed random zoom factors >0, <1 (e.g. 0.3)
	"""

	def __init__(self,zoomRange=0.3, parameters=None):
		super().__init__(str(type(self).__name__))
		if (parameters is None):
			self.parameters['zoomRange'] = zoomRange

		else:
			self.parameters = parameters


	def cropImagePair(self, inputImg, outputMask, patchSize):
		zoomRange = self.parameters['zoomRange']
		zoomFactor = tf.random.uniform((1,1), -zoomRange, zoomRange)
		zoomedNormCrop = [patchSize[0]*(1+zoomFactor[0,0])/inputImg.shape[0],patchSize[1]*(1+zoomFactor[0,0])/inputImg.shape[1]]

		offset = tf.concat([tf.random.uniform((1, 1), 0, 1-zoomedNormCrop[0]),tf.random.uniform((1, 1), 0, 1-zoomedNormCrop[1])],axis=1)
		boxes = tf.concat([offset, offset+zoomedNormCrop], axis=1)
		cropped_image = tf.image.crop_and_resize(tf.expand_dims(inputImg, 0), boxes, [0], patchSize, method='bilinear')
		cropped_annotation = tf.image.crop_and_resize(tf.expand_dims(outputMask, 0), boxes, [0], patchSize, method='nearest')
		return cropped_image,cropped_annotation
