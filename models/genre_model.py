from keras.models import Input, Model
from keras.models import Dense, Conv2D, AveragePooling2D, Dropout

"""
Builds a simple convnet for music information retreival out of a spectrogram.
"""

def build_model(height, width):
    inputLayer = Input((1,height, width))
    convLayer1 = Conv2D(50, (Rows, 1), activation='relu', name='conv1')(inputLayer)      
    convLayer2 = Conv2D(50, (1, 4), activation='relu', name='conv2')(convLayer1)  
    convLayer3 = Conv2D(50, (1, 4), activation='relu', name='conv3')(convLayer2)          
    avgLayer = AveragePooling2D((1, Columns - 6))(convLayer3)          
    
    flatten = Flatten()(avgLayer)        
    dense = Dense(100, activation='relu', name='dense1')(flatten)    
    dense = Dropout(0.3)(dense)
    dense = Dense(OutputSize, activation='softmax', name='dense2')(dense)
    model = Model(inputs=inputLayer, outputs=dense)
    
    return model