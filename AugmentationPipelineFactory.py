from RandomPosZoomCropper import RandomPosZoomCropper
from GaussianNoise import GaussianNoise
from ElasticTransformer import ElasticTransformer
from standardPreprocessors import RandomBrightness, RandomContrast, tfPreprocessing
from BlurKernelPreprocessing import BlurKernelPreprocessing

import json



class AugmentationPipelineFactory():

    def createPipelineFromJSON(self, jsonObject):
        identifier = jsonObject['identifier']
        return globals()[identifier](parameters=jsonObject['parameters'])