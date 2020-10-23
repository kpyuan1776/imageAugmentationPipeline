from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, SpatialDropout2D

from PIL import Image
import io

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Add, Conv2DTranspose, concatenate, AveragePooling2D
import numpy as np
import os
import tensorflow as tf


""" getModel() returns a LinkNet type deep neural network
"""



def firstBlock(i, numFilter, kernelSize=7, poolSize=3, strides=2, initializer='he_uniform', kernelRegularizer=None, layername='initial'):
    x = Conv2D(numFilter,
               kernel_size=kernelSize,
               strides=strides,
               padding='same',
               kernel_initializer=initializer,
               kernel_regularizer=kernelRegularizer,
               name=layername+'_conv2d_1')(i)
    x = BatchNormalization(name=layername+'_bn')(x)
    x = Activation('relu', name=layername+'_relu')(x)
    x = MaxPool2D(poolSize, strides=strides, padding='same',
                  name=layername+'_pool')(x)
    return x


def residualEncoderBlock(i, numFilter, kernelSize=3, strides=1, initializer='he_uniform', kernelRegularizer=None, layername='residualBlock', spatialDropout=0.2):
    x = Conv2D(numFilter,
               kernel_size=kernelSize,
               strides=strides,
               kernel_initializer=initializer,
               kernel_regularizer=kernelRegularizer,
               padding='same',
               name=layername+'_conv2d_1')(i)
    x = BatchNormalization(name=layername+'_bn_1')(x)
    x = Activation('relu', name=layername+'_relu_1')(x)
    x = Conv2D(numFilter,
               kernel_size=kernelSize,
               strides=1,
               kernel_initializer=initializer,
               kernel_regularizer=kernelRegularizer,
               padding='same',
               name=layername+'_conv2d_2')(x)
    #x = Dropout(dropout, name=layername+'_dropout')(x)
    if spatialDropout:
        x = SpatialDropout2D(
            spatialDropout, name=layername+'_spatial_dropout')(x)
    # skip connection
    res = i
    if strides != 1 or tf.keras.backend.int_shape(i)[-1] != numFilter:
        # make skip connection compatible with convolution outputs for summation
        res = Conv2D(numFilter,
                     kernel_size=1,
                     strides=strides,
                     kernel_initializer=initializer,
                     kernel_regularizer=kernelRegularizer,
                     padding='same',
                     name=layername+'_conv2d_res')(i)
    x = Add(name=layername+'_add')([x, res])
    x = BatchNormalization(name=layername+'_bn_2')(x)
    x = Activation('relu', name=layername+'_out')(x)
    return x


def encoderBlock(x, numFilter, kernelSize=3, strides=2, kernelRegularizer=None, spatialDropout=None, name='encoder'):
    x = residualEncoderBlock(x, numFilter, kernelSize=3, strides=strides, kernelRegularizer=kernelRegularizer,
                             layername=name+'residualBlock1', spatialDropout=spatialDropout)
    x = residualEncoderBlock(x, numFilter, kernelSize=3, strides=1, kernelRegularizer=kernelRegularizer,
                             layername=name+'residualBlock2', spatialDropout=spatialDropout)
    return x


def decoderBlock(x, numFilter, kernelSize=3, strides=2, spatialDropout=None, initializer='he_uniform', kernelRegularizer=None, layername='decoder'):
    x = Conv2D(numFilter, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer,
               kernel_regularizer=kernelRegularizer, name=layername+'_conv2d_1')(x)
    x = BatchNormalization(name=layername+'_bn_1')(x)
    x = Activation('relu', name=layername+'_relu_1')(x)
    x = Conv2DTranspose(numFilter,
                        kernel_size=kernelSize,
                        strides=strides,
                        padding='same',
                        kernel_initializer=initializer,
                        name=layername+'_conv2d_transpose')(x)
    x = BatchNormalization(name=layername+'_bn_2')(x)
    x = Activation('relu', name=layername+'_relu_2')(x)
    x = Conv2D(numFilter,
               kernel_size=1,
               strides=1,
               padding='same',
               kernel_initializer=initializer,
               kernel_regularizer=kernelRegularizer,
               name=layername+'_conv2d_2')(x)
    x = BatchNormalization(name=layername+'_bn_3')(x)
    # x = tf.keras.layers.Dropout(dropout, name=layername+'_dropout')(x)
    if spatialDropout:
        x = SpatialDropout2D(
            spatialDropout, name=layername+'_spatial_dropout')(x)
    x = Activation('relu', name=layername+'_relu_3')(x)
    return x


def lastBlock(i, numFilter, out_dim=1, finalActivation=None, kernelSize=3, strides=2,
              initializer='he_uniform', kernelRegularizer=None, layername='final'):
    x = Conv2DTranspose(numFilter,
                        kernel_size=kernelSize,
                        strides=strides,
                        padding='same',
                        kernel_initializer=initializer,
                        kernel_regularizer=kernelRegularizer,
                        name=layername+'_conv2dtranspose_1')(i)
    x = BatchNormalization(name=layername+'_bn_1')(x)
    x = Activation('relu', name=layername+'_relu_1')(x)
    x = Conv2D(numFilter, kernel_size=kernelSize,
               padding='same',
               kernel_initializer=initializer,
               kernel_regularizer=kernelRegularizer,
               name=layername+'_conv2d_1')(x)
    x = BatchNormalization(name=layername+'_bn_2')(x)
    x = Activation('relu', name=layername+'_relu_2')(x)
    x = Conv2DTranspose(out_dim,
                        activation=finalActivation,
                        kernel_size=kernelSize,
                        strides=strides,
                        padding='same',
                        kernel_initializer=initializer,
                        name=layername+'_conv2d_transpose_2')(x)
    return x




def getModel():
    i = Input(shape=[None, None, 1])
    weight_decay = 1e-5  # None
    regularizer = tf.keras.regularizers.l2(weight_decay) if weight_decay else None

    # pdb.set_trace()

    numOfFilters = [4, 16, 48, 144]
    do_multiscale = True

    x = firstBlock(i, numOfFilters[0], kernelRegularizer=regularizer)


    if do_multiscale:
        x_multiscale = AveragePooling2D(pool_size=(4, 4))(i)
        x = concatenate([x_multiscale, x])

    #encoder1 = encoderBlock(x,numOfFilters[1],kernelSize=3,strides=2,kernelRegularizer=regularizer,spatialDropout=None,name='encoder1')
    encoder1 = encoderBlock(x, numOfFilters[1], kernelSize=3, strides=2,
                        kernelRegularizer=regularizer, spatialDropout=0.2, name='encoder1')

    if do_multiscale:
        x_multiscale = AveragePooling2D(pool_size=(2, 2))(x_multiscale)
        x = concatenate([x_multiscale, encoder1])
        #encoder2 = encoderBlock(x,numOfFilters[2],kernelSize=3,strides=2,kernelRegularizer=regularizer,spatialDropout=None,name='encoder2')
        encoder2 = encoderBlock(x, numOfFilters[2], kernelSize=3, strides=2,
                            kernelRegularizer=regularizer, spatialDropout=0.2, name='encoder2')
    else:
        encoder2 = encoderBlock(encoder1, numOfFilters[2], kernelSize=3, strides=2,
                            kernelRegularizer=regularizer, spatialDropout=None, name='encoder2')

    if do_multiscale:
        x_multiscale = AveragePooling2D(pool_size=(2, 2))(x_multiscale)
        x = concatenate([x_multiscale, encoder2])
        #x = encoderBlock(x,numOfFilters[3],kernelSize=3,strides=2,kernelRegularizer=regularizer,spatialDropout=None,name='encoder3')
        x = encoderBlock(x, numOfFilters[3], kernelSize=3, strides=2,
                     kernelRegularizer=regularizer, spatialDropout=0.1, name='encoder3')

    else:
        x = encoderBlock(encoder2, numOfFilters[3], kernelSize=3, strides=2,
                     kernelRegularizer=regularizer, spatialDropout=None, name='encoder3')

    #x = decoderBlock(x,numOfFilters[2], kernelSize=3, strides=2,spatialDropout=None, initializer='he_uniform', kernelRegularizer=regularizer, layername='decoder3')
    x = decoderBlock(x, numOfFilters[2], kernelSize=3, strides=2, spatialDropout=0.1,
                 initializer='he_uniform', kernelRegularizer=regularizer, layername='decoder3')

    x = Add()([encoder2, x])

    #x = decoderBlock(x,numOfFilters[1], kernelSize=3, strides=2,spatialDropout=None, initializer='he_uniform', kernelRegularizer=regularizer, layername='decoder2')
    x = decoderBlock(x, numOfFilters[1], kernelSize=3, strides=2, spatialDropout=0.2,
                 initializer='he_uniform', kernelRegularizer=regularizer, layername='decoder2')
    x = Add()([encoder1, x])
    #x = decoderBlock(x,numOfFilters[1], kernelSize=3, strides=2,spatialDropout=None, initializer='he_uniform', kernelRegularizer=regularizer, layername='decoder1')
    x = decoderBlock(x, numOfFilters[1], kernelSize=3, strides=2, spatialDropout=0.2,
                 initializer='he_uniform', kernelRegularizer=regularizer, layername='decoder1')
    x = lastBlock(x, numOfFilters[0], out_dim=1, finalActivation='sigmoid',
              kernelRegularizer=regularizer, layername='final')

    model = Model(i, x)
    return model
