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

	def __init__(self, inputProcessings: [] = [], imagePatchCropper, imagePairAugmentations: [] = [], patchSize = [256,256], parameters = None, checkSteps=False):
		super().__init__(str(type(self).__name__), parameters)
		self.checkImageStackPreProc=[]
        self.checkImageStackPair=[]
        self.checkSteps = checkSteps
        self.patchSize = patchSize
		if (parameters is None):
			self.inputProcessings = inputProcessings
            self.imagePatchCropper = imagePatchCropper
            self.imagePairAugmentations = imagePairAugmentations
		else:
			self.parameters = parameters
			self.inputProcessings = []
            self.imagePairAugmentations = []
            for p in parameters['inputProcessing']:
                self.inputProcessings.append(factory.createPipelineFromJSON(p))

            self.imagePatchCropper = factory.createPipelineFromJSON(parameters['imagePatchCropper'])

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

        img, annotation = self.imagePatchCropper(image,mask,[self.patchSize,self.patchSize])

        for f in self.imagePairAugmentations:
            imagePatch, maskPatch = f.modifyImagePair(imagePatch, maskPatch)

        return imagePatch, maskPatch



    def loadImage(fpath,num_channels=3):
        image = tf.io.read_file(fpath)
        image = tf.io.decode_png(image, channels=num_channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image




