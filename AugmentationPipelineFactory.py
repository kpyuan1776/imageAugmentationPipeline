from RandomPosZoomCropper import RandomPosZoomCropper
from GaussianNoise import GaussianNoise
from ElasticTransformer import ElasticTransformer
from standardPreprocessors import RandomBrightness, RandomContrast, tfPreprocessing

import json



class ImageAnalysisFactory():

    def createPipelineFromJSON(self, jsonObject):
        identifier = jsonObject['identifier']
        return globals()[identifier](parameters=jsonObject['parameters'])