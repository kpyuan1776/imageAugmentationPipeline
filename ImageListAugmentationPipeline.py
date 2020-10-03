import JSONSerializable
from AugmentationPipelineFactory import AugmentationPipelineFactory
import numpy as np
import json
import collections
from copy import deepcopy

import tensorflow as tf
import random

factory = AugmentationPipelineFactory()


class ImageListAugmentationPipeline(JSONSerializable.JSONSerializable):
    """ Defines a data augmentation pipeline consisting of a list of image operations.
        The pipeline is returned as a generator to use in tf/keras fit() (or previously fit_generator()) pipeline parameters can be read from JSON, stored to JSON

        To use:
        >>> imgaugPipeline = ImageListAugmentationPipeline([inputOperationA, inputOperationB],[operationA(param1,param2,...), operationB(...)])
        >>> dataGenerator = imgaugPipeline.getDataGenerator(imgsList_Input,imgsList_Output)
    """

	def __init__(self, inputProcessings: [] = [], imagePairAugmentations: [] = [],zoomRange = 0.3, patchSize = [256,256], parameters = None, checkSteps=False):
		super().__init__(str(type(self).__name__), parameters)
		self.checkImageStackPreProc=[]
        self.checkImageStackPair=[]
        self.checkSteps = checkSteps
        self.patchSize = patchSize
		if (parameters is None):
			self.inputProcessings = inputProcessings
            self.imagePairAugmentations = imagePairAugmentations
            self.parameters['zoomRange'] = zoomRange
		else:
			self.parameters = parameters
			self.inputProcessings = []
            self.imagePairAugmentations = []
            for p in parameters['inputProcessing']:
                self.inputProcessings.append(factory.createPipelineFromJSON(p))
			for f in parameters['imagePairAugmentation']:
				self.imagePairAugmentations.append(factory.createPipelineFromJSON(f))



	@classmethod
	def fromJSON(cls, jsonString: str):
		data = json.loads(jsonString)
		return cls(parameters=data['parameters'])


	def toJSON(self):
		return {
			'identifier': self.identifier,
			'parameters': {
                'inputProcessing': [
					p.toJSON() for p in self.inputProcessings
				],
				'imagePairAugmentation': [
					f.toJSON() for f in self.imagePairAugmentations
				]
			}
		}





	def getDataGenerator(self,inputList,outputList,batchSize):
		self.resImg = []
		self.parameters['analyzers'] = collections.defaultdict()
		
        if(isinstance(inputList, list) and isinstance(outputList, list)):
            pass
        else:
            raise TypeError('input and output must be list of file paths of image')

        indexlist = list(range(0,len(inputList)))
        random.shuffle(indexlist)
        batchShift = 0
        while(True):
            image = np.zeros((batchSize, patchSize, patchSize, numChannels)).astype('float')
            mask = np.zeros((batchSize, patchSize, patchSize, 1)).astype('float')

            for i in range(batchShift,batchShift+batchSize):
                imgTrain = loadImage(imageList[indexlist[i]],numChannels)
                maskTrain = loadImage(maskList[indexlist[i]],1)

                imgPatch,maskPatch = getAugmentedImageAnnotation(imgTrain,maskTrain,patchSize)

                image[i-batchShift] = imgPatch
                mask[i-batchShift] = maskPatch

            batchShift+=batchSize
            if(batchShift+batchSize >=len(indexlist)):
                batchShift = 0
                random.shuffle(indexlist)
            
            yield tf.cast(image, tf.float32),tf.cast(mask, tf.float32)
		



    def getAugmentedImageAnnotation(self,image,mask,patchSize):
        for p in self.inputProcessings:
            image = p.preprocessImage(image)
            if self.checkSteps:
                self.checkImageStackPreProc.append(image)

        img, annotation = self.getRandomPositionZoomCrop(image,mask,[self.patchSize,self.patchSize])

        for f in self.imagePairAugmentations:
            imagePatch, maskPatch = f.modifyImagePair(imagePatch, maskPatch)

        return imagePatch, maskPatch



    def loadImage(fpath,num_channels=3):
        image = tf.io.read_file(fpath)
        image = tf.io.decode_png(image, channels=num_channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image


    def getRandomPositionZoomCrop(img,mask,crop=[512,512]):
        """ get a random crop with random zoom level of a fixed height and width defined by crop 
            get a box for cropping in normalized image coordinates (i.e. whole image is [0,0,1,1] 
            * pick a random variables $zoomFactor \in  (-zoomRange,+zoomRange)$
            * compute crop in normalized image coordinates
            * compute zoomedCrop in normalized image coordinates
            * pick a tuple of random variables $offset \in ([0,1-zoomedCrop[0]],[0,1-zoomedCrop[1]])$
            * finally crop box is tf.concat([offset, offset+zoomedNormCrop], axis=1)
            ARGS:
                zoomRange: width of interval for allowed random zoom factors >0, <1 (e.g. 0.3) 
        """
        zoomRange = self.parameters['zoomRange']
        zoomFactor = tf.random.uniform((1,1), -zoomRange, zoomRange)
        zoomedNormCrop = [crop[0]*(1+zoomFactor[0,0])/img.shape[0],crop[1]*(1+zoomFactor[0,0])/img.shape[1]]
        
        offset = tf.concat([tf.random.uniform((1, 1), 0, 1-zoomedNormCrop[0]),tf.random.uniform((1, 1), 0, 1-zoomedNormCrop[1])],axis=1)
        boxes = tf.concat([offset, offset+zoomedNormCrop], axis=1)
        cropped_image = tf.image.crop_and_resize(tf.expand_dims(img, 0), boxes, [0], crop, method='bilinear')
        cropped_annotation = tf.image.crop_and_resize(tf.expand_dims(mask, 0), boxes, [0], crop, method='nearest')
        return cropped_image,cropped_annotation


