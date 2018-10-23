import numpy as np
import random

"""
Splits the dataset (data=(x_data, y_data)) into two parts of ratio % / 100% - ratio %.
"""
def split_dataset(data, ratio, seed = -1):
    if (seed == -1):
        seed = random.randint(0, 1000) 

    (x_data, y_data) = data   

    number_of_samples = data[0].shape[0]
    number_of_samples_1 = int(number_of_samples * ratio)
    number_of_samples_2 = number_of_samples - number_of_samples_1

    shape_x_1 = x_data.shape[1:]
    shape_y_1 = y_data.shape[1:]
    shape_x_2 = x_data.shape[1:]
    shape_y_2 = y_data.shape[1:]
    shape_x_1 = (number_of_samples_1, ) + shape_x_1
    shape_y_1 = (number_of_samples_1, ) + shape_y_1
    shape_x_2 = (number_of_samples_2, ) + shape_x_2
    shape_y_2 = (number_of_samples_2, ) + shape_y_2

    x_data_1 = np.zeros(shape_x_1)
    y_data_1 = np.zeros(shape_y_1)
    x_data_2 = np.zeros(shape_x_2)
    y_data_2 = np.zeros(shape_y_2)

    indices = list(range(number_of_samples))
    random.Random(seed).shuffle(indices)

    for i in range(number_of_samples_1):
        x_data_1[i] = x_data[indices[i]]
        y_data_1[i] = y_data[indices[i]]

    for i in range(number_of_samples_1, number_of_samples_2):
        x_data_1[i] = x_data[indices[i]]
        y_data_1[i] = y_data[indices[i]]

    return ((x_data_1, y_data_1), (x_data_2, y_data_2))