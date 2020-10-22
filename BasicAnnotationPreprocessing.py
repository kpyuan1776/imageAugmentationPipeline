import numpy as np
import collections
import JSONSerializer

class BasicAnnotationPreprocessing(JSONSerializer.JSONSerializer):
    def __init__(self, identifier: str, parameters = None):
        super().__init__(identifier, parameters)



    def preprocessAnnotation(self, mask: np.ndarray):
        raise NotImplementedError