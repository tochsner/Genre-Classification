from keras.models import Input, Model
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Dropout

"""
Builds a simple convnet for music information retreival out of a spectrogram.
"""

def build_model(input_shape, output_lenght):
    height = input_shape[0]
    width = input_shape[1]

    inputLayer = Input(input_shape)
    convLayer1 = Conv2D(50, (height, 1), activation='relu', name='conv1')(inputLayer)      
    convLayer2 = Conv2D(50, (1, 4), activation='relu', name='conv2')(convLayer1)  
    convLayer3 = Conv2D(50, (1, 4), activation='relu', name='conv3')(convLayer2)          
    avgLayer = AveragePooling2D((1, width - 6))(convLayer3)          
    
    flatten = Flatten()(avgLayer)        
    dense = Dense(100, activation='relu', name='dense1')(flatten)    
    dense = Dropout(0.3)(dense)
    dense = Dense(output_lenght, activation='softmax', name='dense2')(dense)
    model = Model(inputs=inputLayer, outputs=dense)
    
    return model