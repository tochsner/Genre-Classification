import random
import os
import numpy as np
from skimage.io import imread, imsave
from keras.models import Sequential
from keras.models import Model
from keras.models import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import pooling
from keras import regularizers
from keras.layers.merge import Concatenate
from keras import optimizers
from keras import initializers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import *
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

os.environ['OMP_NUM_THREADS'] = str(8)

SonoPath = "/home/tobia/Documents/ML/Data MA Sono/"

#Costs

class MeanSquareCostFunction:
    def getCost(self, outputValues, correctValues):
        return np.sum(np.power(outputValues - correctValues, 2))
    def getIndividualCost(self, outputValues, correctValues):
        return np.power(outputValues - correctValues, 2)    
    def getDerivatives(self, outputValues, correctValues):
        return outputValues - correctValues

MeanSquareCost = MeanSquareCostFunction()

Rows = 300
Columns = 40
TotalColumns = 400

def importPlaylists():
    with open("/home/tobia/Documents/ML/Genre-Classification/data/genres.csv","r") as f:
        return [np.array(x.strip().split(','))[1:] for x in f.readlines()]
def loadSong(id):
    return imread(SonoPath + str(id) + ".png").ravel() / 256
def getSlice(id):
    start = random.randint(0, TotalColumns - 1 - Columns)
    return np.power(imread(SonoPath + str(id) + ".png")[513 - Rows:,start:start + Columns].reshape(1,Rows, Columns) / 256, 3)
def getAllSlices(id):    
    image = np.power(imread(SonoPath + str(id) + ".png") / 256, 3)          
    return [image[513 - Rows:, x:x+Columns].reshape(1,Rows, Columns) for x in range(0, TotalColumns + 1 -Columns, Columns // 2)]

#load playlists
playlists = importPlaylists()[:50]
playlistIndexes = list(range(len(playlists)))

def getOneHot(index):
    vector = np.zeros((len(playlists)))
    vector[index] = 1
    return vector

def getSamples(count, split):
    times = (count // sum([len(p) for p in playlists]))
    x = np.zeros((times * sum([len(p) for p in playlists]),1, Rows, Columns))
    y = np.zeros((times * sum([len(p) for p in playlists]), len(playlists)))    
    print(sum([len(p) for p in playlists]))
    xTest = np.zeros((sum([len(p) for p in playlists]), TotalColumns // Columns * 2 - 1,1, Rows, Columns))
    yTest = np.zeros((sum([len(p) for p in playlists]), len(playlists)))
    zTest = []
    trainSongsCount = 0
    testSongsCount = 0
    i = 0    
    z = 0
    for p in playlistIndexes:                                                                  
        for s in playlists[p]: 
            if random.random() > split:
                for t in range(times):
                    try:
                        x[i,:,:] = getSlice(s)
                        y[i,:] = getOneHot(p)
                        i += 1
                    except:
                        pass
                trainSongsCount += 1
            else:
                try:
                    xTest[z,:,:,:] = getAllSlices(s)
                    yTest[z,:] = getOneHot(p)
                    zTest.append(s)                    
                    z += 1
                except:
                    pass
                testSongsCount += 1
    x = x[:i]
    y = y[:i]    
    xTest = xTest[:z]
    yTest = yTest[:z]
    zTest = zTest[:z]    

    print(str(i) + " " + str(z))

    return ((x,y), (xTest, yTest, zTest))

def learn(lr = 0.08, dataSize = 7000, path = ""):  
    Max = 0

    OutputSize = len(playlists)
    BatchSize = 10
    LearningRate = lr
    
    sgd = optimizers.SGD(lr=LearningRate)

    inputLayer = Input((1,Rows, Columns))
    convLayer1 = Conv2D(50, (Rows, 1), activation='relu', name='conv1')(inputLayer)      
    convLayer2 = Conv2D(50, (1, 4), activation='relu', name='conv2')(convLayer1)  
    convLayer3 = Conv2D(50, (1, 4), activation='relu', name='conv3')(convLayer2)          
    avgLayer = AveragePooling2D((1, Columns - 6))(convLayer3)          
    flatten = Flatten()(avgLayer)        
    dense = Dense(100, activation='relu', name='dense1')(flatten)    
    dense = Dropout(0.3)(dense)
    dense = Dense(OutputSize, activation='softmax', name='dense2')(dense)
    model = Model(inputs=inputLayer, outputs=dense)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])       
    
    def validate(samples, printErrors = False):
        rightCount = 0
        sliceRightCount = 0

        for i in range(len(samples[0])):
            outputs = model.predict(samples[0][i])            
            
            if np.argmax(np.mean(outputs, axis=0)) == np.argmax(samples[1][i]):
                rightCount += 1   
                if printErrors == True:   
                    print(str(MeanSquareCost.getCost(outputs, samples[1][i])) + " r " + str(samples[2][i]) + " " + str(np.argmax(np.mean(outputs, axis=0))) + " " + str(np.argmax(samples[1][i])))
               
            elif printErrors == True:
                print(str(MeanSquareCost.getCost(outputs, samples[1][i])) + " f " + str(samples[2][i]) + " " + str(np.argmax(np.mean(outputs, axis=0))) + " " + str(np.argmax(samples[1][i])))
               
            if np.argmax(outputs[np.random.randint((400-Columns) // Columns)]) == np.argmax(samples[1][i]):
                sliceRightCount += 1
        return (rightCount / len(samples[0]), sliceRightCount / len(samples[0]))   
                
    print("get Samples", flush=True)
    
    samples = getSamples(dataSize, 0.15)

    print("validate Samples " + str(len(samples[1][0])), flush=True)

    result = validate(samples[1])
    print(str(result[0]) + " " + str(result[1]), flush=True)       

    print("Start with lr " + str(LearningRate), flush=True)
    for i in range(150):        
        model.fit(samples[0][0], samples[0][1], epochs=5, verbose=0, batch_size=BatchSize)
        
        result = validate(samples[1])
        print(str(result[0]) + " " + str(result[1]), flush=True)        
        if result[0] > 0.3 and result[0] > Max:
            Max = result[0]
            validate(samples[1], False)  
        
            model.save_weights(path)                   

    print("----" + str(Max))    

learn(1, 500, "/home/tobia/Documents/Test")     

