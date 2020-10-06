
import numpy as np
import tensorflow as tf
import cv2

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import random


import BasicImagePairAugmentation


class ElasticTransformer(BasicImagePairAugmentation.BasicImagePairAugmentation):
	""" computes an elastic deformation simultaneous for a pair of image and mask
		Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
		and https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation 

		TODO: replace ndimage.gaussian_filter(image, sigma) by blur_size = int(4*sigma) | 1; cv2.GaussianBlur(image, ksize=(blur_size, blur_size), sigmaX=sigma)
		TODO: generalize for image.shape = shape1,shape2,numchannels (colorized image)
	"""

	def __init__(self,alpha=1.5, sigma=0.07, alpha_affine=0.03, random_state=None, parameters=None):
		super().__init__(str(type(self).__name__))
		if random_state is None:
			self.random_state = np.random.RandomState(None)
		if (parameters is None):
			self.parameters['alpha'] = alpha
			self.parameters['sigma'] = sigma
			self.parameters['alpha_affine'] = alpha_affine

		else:
			self.parameters = parameters


	def modifyImagePair(self, inputImg, outputImg):

		mergedImage = np.concatenate((inputImg[..., None], outputImg[..., None]), axis=2)

		shapeImg = mergedImage.shape
		shape_size = shapeImg[:2]

		alpha = shapeImg[1]*self.parameters['alpha']
		sigma = shapeImg[1] * self.parameters['sigma']
		alpha_affine = shapeImg[1] * self.parameters['alpha_affine']

		center_square = np.float32(shape_size) // 2
		square_size = min(shape_size) // 3
		pts1 = np.float32([center_square + square_size, [center_square[0]+square_size,
							center_square[1]-square_size], center_square - square_size])
		pts2 = pts1 + self.random_state.uniform(-alpha_affine,
							alpha_affine, size=pts1.shape).astype(np.float32)
		M = cv2.getAffineTransform(pts1, pts2)

		mergedImage = cv2.warpAffine(mergedImage, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

		dx = gaussian_filter((self.random_state.rand(*shapeImg) * 2 - 1), sigma) * alpha
		dy = gaussian_filter((self.random_state.rand(*shapeImg) * 2 - 1), sigma) * alpha
		dz = np.zeros_like(dx)

		x, y, z = np.meshgrid(np.arange(shapeImg[1]), np.arange(shapeImg[0]), np.arange(shapeImg[2]))
		indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx,(-1, 1)), np.reshape(z, (-1, 1))

		mergedImageWarped = map_coordinates(mergedImage, indices, order=1, mode='reflect').reshape(shapeImg)

		return tf.cast(np.expand_dims(mergedImageWarped[..., 0], axis=-1), tf.float32), tf.cast(np.expand_dims(mergedImageWarped[..., 1], axis=-1), tf.float32)


		
