import json


class ImageAnalysisFactory():

    def createPipelineFromJSON(self, jsonObject):
        identifier = jsonObject['identifier']
        return globals()[identifier](parameters=jsonObject['parameters'])