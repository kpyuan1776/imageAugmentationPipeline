import numpy as np
import collections
import JSONSerializer

class BasicInputPreprocessing(JSONSerializer.JSONSerializer):
    def __init__(self, identifier: str, parameters = None):
        super().__init__(identifier, parameters)



    def preprocessImage(self, img: np.ndarray):
        raise NotImplementedError