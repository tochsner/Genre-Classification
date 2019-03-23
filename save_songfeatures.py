import random
import numpy
import csv
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers.merge import Concatenate
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras import initializers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from itertools import chain
from PIL import Image

import sys

sys.path.insert(0, 'C:/Users/Tobia/Documents/AI/MNIST Experiments')

SonoPath = "C:/Users/Tobia/Documents/AI/Data MA Sono/"


def importPlaylists(path="C:/Users/Tobia/Documents/AI/Music Experiment/SpotifyData/Playlists.csv"):
    with open(path, "r") as f:
        return [np.array(x.strip().split(','))[1:] for x in f.readlines()]


def loadSongFeatures(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        return list(reader)


def loadSong(id):
    return imread(SonoPath + str(id) + ".png").ravel() / 256


def getSlice(id):
    start = random.randint(0, 399 - Columns)
    return np.power(
        imread(SonoPath + str(id) + ".png")[513 - Rows:, start:start + Columns].reshape(1, Rows, Columns) / 256, 3)


def getAllSlices(id):
    image = Image.open(SonoPath + str(id) + ".png")
    array = np.power(np.array(image) / 256, 3)
    return [array[513 - Rows:, x:x + Columns].reshape(1, Rows, Columns) for x in range(0, 401 - Columns, Columns)]


Rows = 300
Columns = 40
OutputSize = 26

def generateFile(inputPath, outputPath):
    baseModel = getBaseModel(inputPath)

    playlists = importPlaylists()
    print("loaded playlists...", flush=True)
    songs = [(p[0], s) for p in playlists for s in p[1:]]
    songsWithoutDuplicates = {d[1]: d for d in songs}
    print(len(songsWithoutDuplicates))
    songs = list(songsWithoutDuplicates.values())

    lines = []

    for i in range(len(songs)):
        try:
            slices = np.array(getAllSlices(songs[i][1]))
            output = np.mean(baseModel.predict_on_batch(slices), axis=0)

            lines.append(
                str(songs[i][0]) + "," + str(songs[i][1]) + "," + ",".join(output[:].astype('str').tolist()) + "\n")
        except:
            pass
        if i % 1000 == 0:
            print(i, flush=True)
            with open(outputPath, "a+") as file:
                file.writelines(lines)
                lines = []

    with open(outputPath, "a+") as file:
        file.writelines(lines)

    print("loaded SongFeatures 1...", flush=True)  