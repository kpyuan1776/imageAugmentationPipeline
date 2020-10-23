import os
import re
import json
import numpy as np

def getListOfImageFoldersFromJson(jsonData,datatype='training'):
    imageList = [] 
    maskList = []
    for imgDataPath in jsonData[datatype]['imagepath']:
        imageList.extend(sortedFileList(imgDataPath))

    for maskDataPath in jsonData[datatype]['maskpath']:
        maskList.extend(sortedFileList(maskDataPath))

    return imageList, maskList



def sortedFileList(filePath):
    fileList = [os.path.join(filePath, f) for f in os.listdir(
        filePath) if re.match(r'.*\.png', f)]
    fileList.sort(key=lambda x: x.split('.png')[0][-4:])
    return fileList


def getMatchingImages(imglist, masklist):
    newMaskList = []
    for idx, item in enumerate(imglist):
        for mask in masklist:
            if mask.split('/')[-1] == item.split('/')[-1]:
                newMaskList.append(mask)

    return imglist, newMaskList



class ImageDataHandler(object):

    def __init__(self,jsonFilePath):
        self._jsonFilePath = jsonFilePath
        self.loadJsonData()


    def loadJsonData(self):
        with open(self._jsonFilePath, "r") as json_file:
            self.jsonData = json.load(json_file)


    def getTrainingData(self):
        imageList, maskList = getListOfImageFoldersFromJson(self.jsonData,datatype='training')
        imageList, maskList = getMatchingImages(imageList, maskList)
        self.numberTrainingData = len(maskList)
        return imageList, maskList

    def getValidationData(self):
        imageList, maskList = getListOfImageFoldersFromJson(self.jsonData,datatype='validation')
        imageList, maskList = getMatchingImages(imageList, maskList)
        self.numberValidationData = len(maskList)
        return imageList, maskList

    def getTestImage(self):
        imageList = self.jsonData['testimage']['imagepath']
        maskList = self.jsonData['testimage']['maskpath']
        return imageList, maskList


    def computeStepsPerEpoch(self,batch_size):
        steps_per_epoch = np.ceil(self.numberTrainingData/batch_size)
        validation_steps = np.ceil(self.numberValidationData/batch_size) 
        return int(steps_per_epoch), int(validation_steps)