import tensorflow as tf
import os
import argparse
import json

import sys
sys.path.append('..')

from metricUtils import CustomIoU, CustomRMSE, CustomPrecision, CustomRecall, CustomMetric
from callbackUtils import TBCallbackImages
import modelUtils 


from RandomPosZoomCropper import RandomPosZoomCropper
from GaussianNoise import GaussianNoise
from ElasticTransformer import ElasticTransformer
from standardPreprocessors import RandomBrightness, RandomContrast, tfPreprocessing
from ImageListAugmentationPipeline import ImageListAugmentationPipeline








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputJson', type=str, required=True,
                        help='json to get all the training data information')
    args = parser.parse_args()

     with open(args.inputJson) as json_file:
        train_set = json.load(json_file)


    with open(os.path.join(datadir,files[0]), "r") as read_file:
        data = json.load(read_file)


    imageList, maskList = getMatchingImages(imageList, maskList)
    print('image and mask list compared: len(img)={} , len(masks)={}'.format(
        len(imageList), len(maskList)))






    model = modelUtils.getModel()
    model.summary()


    model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', CustomMetric(1, 0.5), CustomPrecision(0.5), CustomRecall(0.5), CustomIoU(), CustomRMSE()])



    p = ImageListAugmentationPipeline([RandomBrightness(), RandomContrast(), tfPreprocessing()],
            RandomPosZoomCropper(),
            [ElasticTransformer(), GaussianNoise()],
            patchSize = [512,512])

    # start training
    batch_size = 16  # 128#64#32

    print('length of training {} and validation {}'.format(
        len(imageList[1::2]), len(imageList[0::2])))
    train_gen = p.getDataGenerator(imageList[1::2], maskList[1::2], batch_size)
    val_gen = p.getDataGenerator(imageList[0::2], maskList[0::2], batch_size)


    steps_per_epoch = np.ceil(len(maskList[1::2])/batch_size) 
    validation_steps = np.ceil(len(maskList[0::2])/batch_size) 
    print('steps_per_epoch = {}, validation_steps = {}'.format(
        steps_per_epoch, validation_steps))

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


    r = model.fit_generator(train_gen, validation_data=val_gen, validation_steps=validation_steps,
			steps_per_epoch=steps_per_epoch, epochs=20,#,5,#150,#25,#60
			callbacks=callbacks,max_queue_size=1,workers=1, use_multiprocessing=False)


    model.save('savedModel.h5')