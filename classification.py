import numpy as np

from data.genres import *
from helper.dataset_tools import *
from models.simple_genre_model import *
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as k

"""
Runs genre classification based on spectrograms with keras.
"""

def train_model(id = 0):
    epochs = 30
    batch_size = 32
    slice_width = 40
    split_ratio = 1
    percetage_of_spectrograms_used = 0.35	

    data_train, data_test = load_data_for_keras(slice_width, split_ratio, percetage_of_spectrograms_used)

    print(data_train[0].shape[0], data_test[0].shape[0])

    dataset_augmentator = get_image_data_generator()

    input_shape = data_train[0].shape[1:]
    output_lenght = data_train[1].shape[1]

    model = build_model(input_shape, output_lenght)
    
    model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'])
    
    model.fit_generator(dataset_augmentator.flow(data_train[0], y = data_train[1], batch_size=batch_size), steps_per_epoch=data_train[0].shape[0] / batch_size,
                        validation_data=data_test, epochs=epochs, verbose=1)

    model.save_weights("augmented_final_" + str(id))

train_model(0)
train_model(1)
train_model(2)
train_model(3)
