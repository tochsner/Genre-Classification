import random
import os
import numpy
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
import sys
sys.path.insert(0, 'C:/Users/Tobia/Documents/AI/MNIST Experiments')
from Classes import *
from logger import *
from datetime import datetime

Logger = Logger("Classification 4")
Logger.log("***")
Logger.log(datetime.now())
Logger.log("***")

os.environ['OMP_NUM_THREADS'] = str(8)

SonoPath = "C:/Users/Tobia/Documents/AI/Data MA Sono/"

#Costs
MeanSquareCost = MeanSquareCostFunction()
TOOCCost = ToocCostFunction()

Rows = 300
Columns = 40
TotalColumns = 400

def importPlaylists():
    with open("C:/Users/Tobia/Documents/AI/Music Experiment/SpotifyData/Genres.csv","r") as f:
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
playlists = importPlaylists()
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

def learn(lr = 0.08, dataSize = 7000, path = "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Small 35 1"):  
    Max = 0

    OutputSize = len(playlists)
    BatchSize = 10
    LearningRate = lr

    Logger.log("LearningRate:" + str(LearningRate))          
    Logger.log("BatchSize:" + str(BatchSize))
    Logger.log("TotalSize:" + str(dataSize))
    Logger.log("LearningRate:" + str(LearningRate))
    Logger.log("Start")
    
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
                    Logger.log(str(MeanSquareCost.getCost(outputs, samples[1][i])) + " r " + str(samples[2][i]) + " " + str(np.argmax(np.mean(outputs, axis=0))) + " " + str(np.argmax(samples[1][i])))
            elif printErrors == True:
                print(str(MeanSquareCost.getCost(outputs, samples[1][i])) + " f " + str(samples[2][i]) + " " + str(np.argmax(np.mean(outputs, axis=0))) + " " + str(np.argmax(samples[1][i])))
                Logger.log(str(MeanSquareCost.getCost(outputs, samples[1][i])) + " f " + str(samples[2][i]) + " " + str(np.argmax(np.mean(outputs, axis=0))) + " " + str(np.argmax(samples[1][i])))
            if np.argmax(outputs[np.random.randint((400-Columns) // Columns)]) == np.argmax(samples[1][i]):
                sliceRightCount += 1
        return (rightCount / len(samples[0]), sliceRightCount / len(samples[0]))   
                
    print("get Samples", flush=True)
    
    samples = getSamples(dataSize, 0.15)

    print("validate Samples " + str(len(samples[1][0])), flush=True)

    result = validate(samples[1])
    print(str(result[0]) + " " + str(result[1]), flush=True)      
    Logger.log(str(result[0]) + " " + str(result[1]))  

    print("Start with lr " + str(LearningRate), flush=True)
    for i in range(150):        
        model.fit(samples[0][0], samples[0][1], epochs=5, verbose=0, batch_size=BatchSize)
        
        result = validate(samples[1])
        print(str(result[0]) + " " + str(result[1]), flush=True)
        Logger.log(str(result[0]) + " " + str(result[1]))
        if result[0] > 0.3 and result[0] > Max:
            Max = result[0]
            validate(samples[1], False)  
        
            model.save_weights(path)                   

    print("----" + str(Max))
    Logger.log("----" + str(Max))

learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 27")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 28")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 29")

learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 30")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 31")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 32")
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 33")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 34")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 35")
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 36")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 37")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 38")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 39")  

learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 40")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 41")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 42")
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 43")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 44")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 45")
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 46")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 47")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 48")     
learn(1, 15000, "C:/Users/Tobia/Documents/AI/Music Experiment/NN/Genre 49")     



Logger.log("***")
Logger.log(datetime.now())
Logger.log("***")