"""
Loads the genre-13 dataset based on Spotify playlists containing 200 songs in 26 different categories.
Data gets formatted for the use with Keras.
"""

import random
import numpy as np
from skimage.io import imread, imsave

spectrogram_path = "data/sonos"
spectrogram_type = ".png"
genres_path = "data/genres.csv"

"""
Imports a list of the songs in each genre. (genre, SpotifyURI)
"""
def load_genres():
    with open(genres_path, "r") as f:
        return [np.array(x.strip().split(','))[1:] for x in f.readlines()]
"""
Loads the spectrogram for a specific Spotify URI (id).
"""
def load_spectrogram(uri):    
    return np.zeros((400,500,1))

    spectrogram = imread(spectrogram_path + "/" + str(uri) + spectrogram_type) / 256

    height = spectrogram.shape[0]
    width = spectrogram.shape[1]        

    return spectrogram.reshape(height, width, 1)
def load_random_slice_of_spectrogram(uri, slice_width):
    spectrogram = load_spectrogram(uri)
    
    height = spectrogram.shape[0]
    width = spectrogram.shape[1]

    start_index = random.randint(0, width - slice_width - 1)

    return spectrogram[:, start_index : start_index + slice_width]
def load_all_slices_of_spectrogram(uri, slice_width):    
    spectrogram = load_spectrogram(uri)
    
    height = spectrogram.shape[0]
    width = spectrogram.shape[1]    

    return [spectrogram[:, start_index : start_index + slice_width] for start_index in range(0, width - slice_width + 1, slice_width)]

def load_data_for_keras(slice_width):     
    genres = load_genres()

    slices_count = 0
    genres_count = len(genres)

    spectrograms = {}

    for genre in genres:
        for song in genre:
            if song not in spectrograms:                               
                try:               
                    all_slices = load_all_slices_of_spectrogram(song, slice_width)
                    spectrograms[song] = all_slices
                    slices_count += len(all_slices)
                except:
                    pass

    height = next(iter(spectrograms.values()))[0].shape[0]
    width = next(iter(spectrograms.values()))[0].shape[1]  

    #uses channels_last
    x_data = np.zeros((slices_count, height, width, 1))
    y_data = np.zeros((slices_count, genres_count))

    i = 0

    for g in range(genres_count):
        for song in genres[g]:
            if song in spectrograms:
                for slice in spectrograms[song]:
                    x_data[i] = slice
                    y_data[i] = np.zeros((genres_count))
                    y_data[i][g] = 1
    
    return (x_data, y_data)