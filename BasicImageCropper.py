import numpy as np
import collections
import JSONSerializable

class BasicImageCropper(JSONSerializable.JSONSerializable):
    def __init__(self, identifier: str, parameters = None):
        super().__init__(identifier, parameters)



    def cropImagePair(self, inputImg, outputImg, patchSize: np.ndarray):
        raise NotImplementedError