import numpy as np

from data.genres import *
from helper.dataset_tools import *
from models.simple_genre_model_hs import *
from helper.hyperparameter_search import HyperparameterSearch
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

"""
Runs genre classification based on spectrograms with keras.
"""

hp_search = HyperparameterSearch("Genre-Classification Hyperparameter-Search Baseline")

epochs = 100
slice_width = 40
split_ratio = 0.7
percentage_of_spectrograms_used = 0.5

early_stopping = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.02, patience=10, restore_best_weights=True)

data_train, data_test = load_data_for_keras(slice_width, split_ratio, percentage_of_spectrograms_used)

input_shape = data_train[0].shape[1:]
output_lenght = data_train[1].shape[1]

def train(params):
    model = build_model(input_shape, output_lenght, params)
    model.compile(loss=params['loss'],
                    optimizer=params['optimizer'](lr=params['lr']),
                    metrics=['accuracy'])

    model.fit(x=data_train[0], y=data_train[1], validation_data=data_test, epochs=epochs, verbose=1, callbacks=[early_stopping])

    return model.evaluate(x=data_test[0], y=data_test[1])

params = {"optimizer": [Adam],
          "loss": ["categorical_crossentropy"],
          "conv_filter": [50],
          "neurons_dense": [100],   
            "lr": [0.0001, 0.000333, 0.001, 0.0333, 0.01],
          "dropout": [0, 0.3]}

hp_search.scan(train, params, 1)
