import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import io



def load_images(fpath, num_channels=3):
    image = tf.io.read_file(fpath)
    image = tf.io.decode_png(image, channels=num_channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def convertToSummary(image):
    height, width, channel = image.shape
    img = Image.fromarray(image)
    output = io.BytesIO()
    img.save(output, format='PNG')  # can I replace that by proper cv2 function
    encodedString = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=encodedString)


def displayImagePair(image, overlayImg,rawimg):
    figure = plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1, title='probability map')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image)
    plt.subplot(1, 3, 2, title='contours')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(overlayImg)
    plt.subplot(1, 3, 3, title='raw image')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(rawimg, cmap='gray')
    return figure


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class TBCallbackImages(tf.keras.callbacks.Callback):

    def __init__(self, imgPath, logDir,  valSteps, patchSize=[512,512]):
        super().__init__()
        self.imagePath = imgPath
        self.logDir = os.path.join(logDir)
        self.valSteps = valSteps
        self.filewriter = tf.summary.create_file_writer(self.logDir+'/overlay')
        self.patchSize = patchSize
        # self._initWriter()

    def getContours(self, ret):
        if len(ret) == 3:
            # opencv 3
            contours = ret[1]
        else:
            # opencv 4
            contours = ret[0]
        return contours

    # def _initWriter(self):
    #  self.filewriter = tf.summary.create_file_writer(self.logDir)

    def on_epoch_end(self, epoch, logs={}):

        imgin = load_images(self.imagePath[0], 1)[0:self.patchSize[0], 0:self.patchSize[1]]
        rawimg = tf.image.per_image_standardization(imgin)
        rawimg = tf.expand_dims(rawimg, axis=0)
        res = self.model.predict(rawimg)
        probabilityMap = (np.squeeze(res)>0.5).astype(np.uint8)*255

        cntOverlayImage = cv2.imread(self.imagePath[0], 1)[0:self.patchSize[0], 0:self.patchSize[1]]
        rawimg = cv2.imread(self.imagePath[0], 1)[0:self.patchSize[0], 0:self.patchSize[1]]

        ret = cv2.findContours(probabilityMap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #ret2 = cv2.findContours(groundtruth, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #cntOverlayImage = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cntOverlayImage, self.getContours(
            ret), -1, (0, 0, 255), 2)
        #cv2.drawContours(cntOverlayImage, self.getContours(ret2), -1, (0, 255, 0), 2)
        #cntOverlayImage[probabilityMap == 255, 1] = 0.8 * cntOverlayImage[probabilityMap == 255, 1]
        #cntOverlayImage[probabilityMap == 255, 2] = 0.8 * cntOverlayImage[probabilityMap == 255, 2]

        figure = displayImagePair(probabilityMap, cntOverlayImage, cv2.cvtColor(rawimg, cv2.COLOR_BGR2GRAY))
        with self.filewriter.as_default():
            tf.summary.image("Training data", plot_to_image(figure), step=epoch)

    def on_train_end(self, logs=None):
        self.filewriter.close()




class OneCycleLearningRatePolicy(tf.keras.callbacks.Callback):
    """cyclical learning rate policy.
    https://arxiv.org/abs/1803.09820
    https://arxiv.org/abs/1708.07120
    https://github.com/fastai/fastai
    """

    def __init__(self, maxLearningRate, endPercentage=0.1, scalePercentage=None,
                    maxMomentum=0.95, minMomentum=0.85,verbose=True):
        super(OneCycleLearningRatePolicy, self).__init__()

        if endPercentage < 0. or endPercentage > 1.:
            raise ValueError('endPercentage not in [0,1]')
        
        if scalePercentage is not None and (scalePercentage < 0. or scalePercentage > 1.):
            raise ValueError('scalePercentage not in [0,1]')

        self.initialLearningRate = maxLearningRate
        self.endPercentage = endPercentage
        self.scale = float(scalePercentage) if scalePercentage is not None else float(endPercentage)
        self.maxMomentum = maxMomentum
        self.minMomentum = minMomentum
        self.verbose = verbose

        if self.maxMomentum is not None and self.minMomentum is not None:
            self._updateMomentum = True
        else:
            self._updateMomentum = False

        self.clrIterations = 0.
        self.history = {}

        self.epochs = None
        self.batch_size = None
        self.samples = None
        self.steps = None
        self.numIterations = None
        self.midCycleId = None


    def _resetCallback(self):
        self.clrIterations = 0.
        self.history = {}


    def computeLearningRate(self):
        if self.clrIterations > 2 * self.midCycleId:
            current_percentage = (self.clrIterations - 2 * self.midCycleId)
            current_percentage /= float((self.numIterations - 2 * self.midCycleId))
            newLearningRate = self.initialLearningRate * (1. + (current_percentage *(1. - 100.) / 100.)) * self.scale
        elif self.clrIterations > self.midCycleId:
            current_percentage = 1. - (self.clrIterations - self.midCycleId) / self.midCycleId
            newLearningRate = self.initialLearningRate * (1. + current_percentage *(self.scale * 100 - 1.)) * self.scale
        else:
            current_percentage = self.clrIterations / self.midCycleId
            newLearningRate = self.initialLearningRate * (1. + current_percentage *(self.scale * 100 - 1.)) * self.scale

        if self.clrIterations == self.numIterations:
            self.clrIterations = 0

        return newLearningRate


    def computeMomentum(self):
        if self.clrIterations > 2 * self.midCycleId:
            newMomentum = self.maxMomentum

        elif self.clrIterations > self.midCycleId:
            current_percentage = 1. - ((self.clrIterations - self.midCycleId) / float(self.midCycleId))
            newMomentum = self.maxMomentum - current_percentage * (self.maxMomentum - self.minMomentum)
        else:
            current_percentage = self.clrIterations / float(self.midCycleId)
            newMomentum = self.maxMomentum - current_percentage * (self.maxMomentum - self.minMomentum)

        return newMomentum




    def on_train_begin(self, logs={}):
        logs = logs or {}

        self.epochs = self.params['epochs']
        self.batch_size = self.params['batch_size']
        self.samples = self.params['samples']
        self.steps = self.params['steps']

        if self.steps is not None:
            self.numIterations = self.epochs * self.steps
        else:
            if (self.samples % self.batch_size) == 0:
                remainder = 0
            else:
                remainder = 1
            self.numIterations = (self.epochs + remainder) * self.samples // self.batch_size

        self.midCycleId = int(self.numIterations * ((1. - self.endPercentage)) / float(2))

        self._resetCallback()
        tf.keras.backend.set_value(self.model.optimizer.lr, self.computeLearningRate())

        if self._updateMomentum:
            if not hasattr(self.model.optimizer, 'momentum'):
                raise ValueError("Momentum can be only updated for SGD optimizer")

            tf.keras.backend.set_value(self.model.optimizer.momentum, self.computeMomentum())



    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}

        self.clrIterations += 1
        newLearningRate = self.computeLearningRate()

        self.history.setdefault('lr', []).append(
            tf.keras.backend.get_value(self.model.optimizer.lr)
            )
        tf.keras.backend.set_value(self.model.optimizer.lr,newLearningRate)

        if self._updateMomentum:
            if not hasattr(self.model.optimizer, 'momentum'):
                raise ValueError("Momentum can be only updated for SGD optimizer")

            newMomentum = self.computeMomentum()

            self.history.setdefault('momentum', []).append(
                tf.keras.backend.get_value(self.model.optimizer.momentum)
            )
            tf.keras.backend.set_value(self.model.optimizer.momentum, newMomentum)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)



    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            if self._updateMomentum:
                print(" - lr: %0.5f - momentum: %0.2f " % (self.history['lr'][-1], self.history['momentum'][-1]))

            else:
                print(" - lr: %0.5f " % (self.history['lr'][-1]))

