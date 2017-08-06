import locale

# Set to users preferred locale:
locale.setlocale(locale.LC_ALL, '')
import numpy as np
import pandas as pd

from datetime import datetime
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dropout, GlobalAveragePooling2D, Dense
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Conv2D, MaxPooling2D
from tqdm import tqdm
import cv2
import invasive
########## Globale Variablen etc. ###################

weights_path = "../weights/groundup_weights.h5"
model_path = '../models/groundup.h5'
img_height = 300
img_width = 400



####################################################################################################

def build_nn(weights_path,img_width,img_height):
        model = Sequential()
        model.add(Conv2D(16, (3,3), activation = 'relu', input_shape=(img_width,img_height,3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(32, (3,3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64, (3,3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(128, (3,3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dropout(0.6))
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(0.6))
        model.add(Dense(1, activation = 'sigmoid'))
        model.load_weights(weights_path)
        optim = optimizers.SGD(lr = 0.0001, decay = 1e-6, momentum = 0.9, nesterov = True)
        model.compile(loss = 'binary_crossentropy',
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])
        return model


model = build_nn(weights_path,img_width,img_height)
model = invasive.train_model(model, weights_path,img_width,img_height)
model.save_weights(weights_path)
model.save(model_path)
predictions = invasive.makepredictions(model)
np.save(open('../predictions/groundup_prediction_array.npy', 'w'), predictions)
