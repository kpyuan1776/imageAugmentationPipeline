import numpy as np
import collections
import JSONSerializer

class BasicImagePairAugmentation(JSONSerializer.JSONSerializer):
    def __init__(self, identifier: str, parameters = None):
        super().__init__(identifier, parameters)



    def modifyImagePair(self, inputImg: np.ndarray, outputImg: np.ndarray):
        raise NotImplementedError