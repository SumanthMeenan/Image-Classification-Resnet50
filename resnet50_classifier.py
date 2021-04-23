import os 
import cv2
import sys
import keras
import constants
from resnet50 import ResNet50
from PIL import * 
import numpy as np
import pandas as pd
# import tensorflow as tf
from keras import optimizers
from keras import applications
from keras import backend as K
from os import listdir, makedirs
from keras.utils.data_utils import Sequence
from os.path import join, exists, expanduser
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    constants.TRAIN_DATA,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle = True)

test_generator = test_datagen.flow_from_directory(
    constants.TEST_DATA,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    constants.VAL_DATA,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical')

def train(train_generator, validation_generator):
    model = Sequential()
    model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
    model.add(Dense(constants.NUM_CLASSES, activation = 'softmax'))
    model.layers[0].trainable = False
    
    sgd = SGD(lr=0.001, decay=5e-6, momentum=0.87, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    model.fit_generator(train_generator,
                        steps_per_epoch=constants.NB_TRAIN_SAMPLES // constants.BATCH_SIZE,
                        epochs=constants.EPOCHS, validation_data=validation_generator,
                        validation_steps=constants.NB_VALIDATION_SAMPLES // constants.BATCH_SIZE,callbacks=[constants.CHECKPOINTER, constants.TENSORBOARD])
    print(" Inside Train")
    model.save_weights(constants.WEIGHTS_PATH)


# def test(test_image):
#     model = build_model()
#     model.load_weights(checkpoint_path)
#     im = cv2.imread(test_image)
#     im = im.reshape([-1, 256, 256, 3])
#     print(model.predict_proba(im))

# def retrain():
#     model = build_model()
#     model.load_weights(checkpoint_path)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['categorical_accuracy'])
#     print(model)
#     model.fit_generator(train_generator,
#                         steps_per_epoch=nb_train_samples // batch_size,
#                         epochs=epochs, validation_data=validation_generator,
#                         validation_steps=nb_validation_samples // batch_size, callbacks=[checkpointer, tensorboard])
#     model.save_weights(checkpoint_path)

# def test_batch(img_folder):
#     model = build_model()
#     model.load_weights(weights_path)

#     # batch_test_datagen = ImageDataGenerator(rescale=1. / 255)

#     # batch_test_generator = batch_test_datagen.flow_from_directory(
#     #     img_folder,
#     #     target_size=(img_width, img_height),
#     #     batch_size=batch_size, class_mode='binary')

#     for img in os.listdir(img_folder):
#         im = cv2.imread(img_folder + img)
#         im = im.reshape([-1, 256, 256, 3])
#         print(img_folder + img, model.predict_proba(im))


if __name__ == "__main__":

    if sys.argv[1] == 'train':
        train(train_generator, validation_generator)
    if sys.argv[1] == 'test':
        test(sys.argv[2])
    if sys.argv[1] == 'test_batch':
        test_batch(sys.argv[2])
    if sys.argv[1] == 'retrain':
        retrain()