import tensorflow as tf
import os
import argparse


import sys
sys.path.append('..')

from metricUtils import CustomIoU, CustomRMSE, CustomPrecision, CustomRecall, CustomMetric
from callbackUtils import TBCallbackImages
import modelUtils 
from dataUtils import ImageDataHandler


from RandomPosZoomCropper import RandomPosZoomCropper
from GaussianNoise import GaussianNoise
from ElasticTransformer import ElasticTransformer
from standardPreprocessors import RandomBrightness, RandomContrast, tfPreprocessing
from ImageListAugmentationPipeline import ImageListAugmentationPipeline




import pdb



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputJson', type=str, required=True,
                        help='json to get all the training data information')
    args = parser.parse_args()
    
    batch_size = 16 # 128#64#32
    numEpoch = 2 #,5,#150,#25,#60
    
    datahandler = ImageDataHandler(args.inputJson)
    imageListTraining, maskListTraining = datahandler.getTrainingData()
    imageListValidation, maskListValidation = datahandler.getValidationData()
    testimages, testmasks = datahandler.getTestImage()

    steps_per_epoch, validation_steps = datahandler.computeStepsPerEpoch(batch_size)

    print('length of training {} and validation {}'.format(
        len(imageListTraining), len(imageListValidation)))

    print('epoch {}, steps training {} and steps validation {}'.format(
        numEpoch, steps_per_epoch, validation_steps))

    validationImgPath = [testimages[0], testmasks[0]]
    

    model = modelUtils.getModel()
    model.summary()


    model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', CustomMetric(1, 0.5), CustomPrecision(0.5), CustomRecall(0.5), CustomIoU(), CustomRMSE()])


    p = ImageListAugmentationPipeline([RandomBrightness(), RandomContrast(), tfPreprocessing()],
            RandomPosZoomCropper(),
            [ElasticTransformer(), GaussianNoise()],
            patchSize = [512,512])

    
    train_gen = p.getDataGenerator(imageListTraining, maskListTraining, batch_size)
    val_gen = p.getDataGenerator(imageListValidation, maskListValidation, batch_size)


    callbacks = [tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('savedModels', 'model.h5'),
        save_best_only=True,
        mode='min',
        verbose=1,
        period=2)]
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join('savedModels', 'logs'),
        histogram_freq=1,
        update_freq="epoch",
        batch_size=batch_size))
    callbacks.append(TBCallbackImages(
        imgPath=validationImgPath,
        logDir=os.path.join('savedModels', 'logs'),
        valSteps=validation_steps))
    # start tensorboard from cmd:  tensorboard --logdisavedModels/logs/
    callbacks.append(tf.keras.callbacks.TerminateOnNaN()
                 )  # terminate if loss is nan
    #  callbacks.append(OneCycleLR(max_lr=0.003,maximum_momentum=None,minimum_momentum=None))

    #pdb.set_trace()

    r = model.fit_generator(train_gen, validation_data=val_gen, validation_steps=validation_steps,
			steps_per_epoch=steps_per_epoch, epochs=numEpoch,
			callbacks=callbacks,max_queue_size=1,workers=1, use_multiprocessing=False)


    model.save('savedModel.h5')
