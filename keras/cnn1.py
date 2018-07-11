import sys
import os

import keras
from keras.preprocessing.image import ImageDataGenerator
import keras.models
import keras.layers as kl

from pudb import set_trace as st
from functools import partial
from pprint import pprint
pp = partial(pprint, width=120, indent=4)

if len(sys.argv) < 1:
    raise ValueError('Please specify data folder containing training, validation and test data!')

## Hyper parameters
hparams = {
    'img_size': (200, 200),
    'batch_size': 32,
    'epochs': 50,
}

data_folder = sys.argv[1]
folders = {x + '_folder': os.path.join(data_folder, x) for x in ['training', 'validation', 'testing']}
hparams.update(folders)

pp('Generating CNN with configuration')
pp(hparams)

## Prepare data

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        hparams['training_folder'],
        target_size=hparams['img_size'],
        batch_size=hparams['batch_size'],
        class_mode='binary')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        hparams['validation_folder'],
        target_size=hparams['img_size'],
        batch_size=hparams['batch_size'],
        class_mode='binary')

## Build model
model = keras.models.Sequential()
model.add(kl.Conv2D(64, (5, 5), strides=(1, 1), padding='valid', input_shape=train_generator.image_shape))
model.add(kl.Activation('relu'))
model.add(kl.Conv2D(32, (3, 3), strides=(1, 1), padding='valid'))
model.add(kl.Activation('relu'))
model.add(kl.Flatten())
model.add(kl.Dense(64))
model.add(kl.Activation('relu'))
model.add(kl.Dropout(0.5))
model.add(kl.Dense(1))
model.add(kl.Activation('sigmoid'))

## Compile
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',  # 'mean_squared_error'
              metrics=['accuracy', 'mean_squared_error'])

model.fit_generator(
        train_generator,
        steps_per_epoch=1000 // hparams['batch_size'],
        epochs=hparams['epochs'],
        validation_data=validation_generator,
        validation_steps=100 // hparams['batch_size'],
        verbose = 1,
)
    
model.save_weights('cnn1_weights.h5')