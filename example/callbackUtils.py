import numpy as np
import cv2
import tensorflow as tf





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
    plt.imshow(rawimg)
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

    def __init__(self, imgPath, logDir,  valSteps):
        super().__init__()
        self.imagePath = imgPath
        self.logDir = os.path.join(logDir)
        self.valSteps = valSteps
        self.filewriter = tf.summary.create_file_writer(self.logDir+'/overlay')
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
        # for i in range(self.valSteps):

        imgin = load_images(self.imagePath[0], 1)[0:512, 0:512]
        rawimg = tf.image.per_image_standardization(imgin)
        rawimg = tf.expand_dims(rawimg, axis=0)#tf.expand_dims(tf.expand_dims(rawimg, axis=0), axis=-1)
        res = self.model.predict(rawimg)
        probabilityMap = (np.squeeze(res)>0.5).astype(np.uint8)*255

        # sumProbabilityMap = tf.Summary(
        #          value=[tf.Summary.Value(tag='val/probmap/{}'.format(i),
        #          image=convertToSummary(probabilityMap))
        #      ])
        #self.filewriter.add_summary(sumProbabilityMap, epoch)

        cntOverlayImage = cv2.imread(self.imagePath[0], 1)[0:512, 0:512]
        rawimg = cv2.imread(self.imagePath[0], 1)[0:512, 0:512]
        #groundtruth = cv2.imread(self.imagePath[1], 0)[0:512, 0:512]
        ret = cv2.findContours(probabilityMap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #ret2 = cv2.findContours(groundtruth, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #cntOverlayImage = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cntOverlayImage, self.getContours(
            ret), -1, (0, 0, 255), 2)
        #cv2.drawContours(cntOverlayImage, self.getContours(ret2), -1, (0, 255, 0), 2)
        cntOverlayImage[probabilityMap == 255, 1] = 0.8 * cntOverlayImage[probabilityMap == 255, 1]
        cntOverlayImage[probabilityMap == 255, 2] = 0.8 * cntOverlayImage[probabilityMap == 255, 2]
        # sumCntImg = tf.Summary(
        #          value=[tf.Summary.Value(tag='val/contour/{}'.format(i),
        #          image=convertToSummary(cntOverlayImage))
        #      ])

        figure = displayImagePair(probabilityMap, cntOverlayImage,rawimg)
        with self.filewriter.as_default():
            tf.summary.image("Training data", plot_to_image(figure), step=epoch)

    def on_train_end(self, logs=None):
        self.filewriter.close()

