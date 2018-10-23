import numpy as np

from data.genres import *
from helper.dataset_tools import *
from models.simple_genre_model import *

"""
Runs genre classification based on spectrograms with keras.
"""

epochs = 10
slice_width = 200
split_ratio = 0.7

data = load_data_for_keras(slice_width)
(data_train, data_test) = split_dataset(data, split_ratio)

input_shape = data_train[0].shape[1:]
output_lenght = data_train[1].shape[1]

model = build_model(input_shape, output_lenght)
model.compile(loss='mse',
                optimizer='adam',
                metrics=['accuracy'])

model.fit(x = data_train[0], y = data_train[1], validation_data=data_test, epochs=epochs)