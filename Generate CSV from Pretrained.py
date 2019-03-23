import random
import numpy as np
import csv
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras import initializers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.utils import np_utils
from keras import backend as k
from itertools import chain
from PIL import Image
SonoPath = "C:/Users/Tobia/Documents/AI/Data MA Sono/"
k.set_image_dim_ordering = "th"

def importPlaylists(path = "C:/Users/tobia/Documents/Programmieren/AI/MusicSimilarity/data/Playlists.csv"):
    with open(path,"r") as f:
        return [np.array(x.strip().split(','))[1:] for x in f.readlines()]
def loadSongFeatures(path):    
    with open(path, 'r') as f:
        reader = csv.reader(f)
        return list(reader)
def loadSong(id):
    return imread(SonoPath + str(id) + ".png").ravel() / 256
def getSlice(id):
    start = random.randint(0, 399 - Columns)
    return np.power(imread(SonoPath + str(id) + ".png")[513 - Rows:,start:start + Columns].reshape(1,Rows, Columns) / 256, 3)
def getAllSlices(id): 
    image = Image.open(SonoPath + str(id) + ".png")   
    array = np.power(np.array(image) / 256, 3)          
    return [array[513 - Rows:, x:x+Columns].reshape(1,Rows, Columns) for x in range(0, 401-Columns, Columns)]

def std_layer(x):
    return k.var(x, axis=2, keepdims=True)

Rows = 263
Columns = 40
OutputSize = 26

def getBaseModel(path):
    input_shape = (Rows, Columns, 1)

    height = input_shape[0]
    width = input_shape[1]

    inputLayer = Input(input_shape)
    convLayer1 = Conv2D(60, (height, 1), activation='relu', name='conv1')(inputLayer)
    convLayer1 = BatchNormalization()(convLayer1)
    convLayer2 = Conv2D(60, (1, 4), activation='relu', name='conv2')(convLayer1)
    convLayer2 = BatchNormalization()(convLayer2)
    convLayer3 = Conv2D(60, (1, 4), activation='relu', name='conv3')(convLayer2)
    convLayer3 = BatchNormalization()(convLayer3)

    avgLayer1 = AveragePooling2D((1, width))(convLayer1)
    avgLayer2 = AveragePooling2D((1, width - 3))(convLayer2)    
    avgLayer3 = AveragePooling2D((1, width - 6))(convLayer3)    
    stdLayer = Lambda(std_layer)(convLayer3)
    maxLayer = MaxPooling2D((1, width - 6))(convLayer3)
    concatenated = Concatenate()([avgLayer3, stdLayer, maxLayer])

    flatten1 = Flatten()(avgLayer1)
    flatten2 = Flatten()(avgLayer2)
    flatten3 = Flatten()(avgLayer3)

    flatten = Flatten()(concatenated)
    dense1 = Dense(120, activation='relu', name='dense1')(flatten)
    dense2 = BatchNormalization()(dense1)
    dense2 = Dense(OutputSize, activation='softmax', name='dense2')(dense2)

    output = Concatenate()([flatten1, flatten2, flatten3, dense1, dense2])

    model = Model(inputs=inputLayer, outputs=output)
    
    model.load_weights(path, by_name=True) 

    return model   
  
def generateFile(inputPath, outputPath):
    baseModel = getBaseModel(inputPath)  

    playlists = importPlaylists()
    print("loaded playlists...", flush=True) 
    songs = [(p[0],s) for p in playlists for s in p[1:]]
    songsWithoutDuplicates = {d[1]:d for d in songs} 
    print(len(songsWithoutDuplicates))   
    songs = list(songsWithoutDuplicates.values())

    lines = []

    for i in range(len(songs)):    
        try:            
            slices = np.array(getAllSlices(songs[i][1]))
            output = np.mean(baseModel.predict_on_batch(slices), axis=0)            

            lines.append(str(songs[i][0]) + "," + str(songs[i][1]) + "," + ",".join(output[:].astype('str').tolist()) + "/n")                              
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

def extendCSV(fileName):
    playlists = importPlaylists()
    allSongs = loadSongFeatures(fileName)

    lines = []

    songs = [(p[0],s) for p in playlists for s in p[1:]]    
    for s1 in songs:
        for s2 in allSongs:
            if s2[1] == s1[1]:
                lines.append(str(s2[0]) + "," + str(s2[1]) + "," + ",".join(s2[2:]) + "/n")                              
        

    with open(fileName, "w+") as file:
        file.writelines(lines) 

generateFile("augmented_final_0", "test.py")
