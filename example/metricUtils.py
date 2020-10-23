
import tensorflow as tf
import numpy as np





class CustomIoU(object):
    """ Intersection over Union
    """

    def __init__(self):
        self.name = 'iou'

    def __name__(self):
        return self.name

    def __call__(self, y_true, y_pred):
        y_t = y_true[:, :, :, 0]
        y_p = tf.cast(tf.greater(y_pred[:, :, :, 0], 0.5), tf.float32)
        intersection = tf.keras.backend.sum(y_t * y_p, axis=[1, 2])
        union = tf.keras.backend.sum(y_t + y_p, axis=[1, 2]) - intersection
        iou = intersection / (union + tf.keras.backend.epsilon())
        return tf.keras.backend.mean(iou, axis=0)


class CustomRMSE(object):
    """ Intersection over Union
    """

    def __init__(self):
        self.name = 'rmse'

    def __name__(self):
        return self.name

    def __call__(self, y_true, y_pred):
        area_true = tf.keras.backend.mean(y_true[:, :, :, 0], axis=[1, 2])
        area_pred = tf.keras.backend.mean(y_pred[:, :, :, 0], axis=[1, 2])
        return tf.keras.backend.sqrt(tf.keras.metrics.mse(area_true, area_pred))


class CustomPrecision(object):
    """ Intersection over Union
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.name = 'precision'

    def __name__(self):
        return self.name

    def __call__(self, y_true, y_pred):
        y_p = tf.cast(tf.greater(
            y_pred[:, :, :, 0], self.threshold), tf.float32)
        tp = tf.keras.backend.sum(y_true[:, :, :, 0] * y_p)
        tp_fn = tf.keras.backend.sum(y_p)
        return tp / (tp_fn + tf.keras.backend.epsilon())


class CustomRecall(object):
    """ Intersection over Union
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.name = 'recall'

    def __name__(self):
        return self.name

    def __call__(self, y_true, y_pred):
        y_p = tf.cast(tf.greater(
            y_pred[:, :, :, 0], self.threshold), tf.float32)
        tp = tf.keras.backend.sum(y_true[:, :, :, 0] * y_p)
        tp_fp = tf.keras.backend.sum(y_true[:, :, :, 0])
        return tp / (tp_fp + tf.keras.backend.epsilon())




class CustomMetric(object):

    def __init__(self, min_distance=1, threshold=0.5):
        self.min_distance = min_distance
        self.threshold = threshold
        self.name = 'somemetric'

    def __name__(self):
        return self.name

    def __call__(self, y_true, y_pred):
        pred_mask = func(y_pred, y_true, self.threshold)

        return pred_mask*self.threshold


def func(a, b, const):
    temp = tf.keras.layers.Add()([a, b])
    return temp