import locale

# Set to users preferred locale:
locale.setlocale(locale.LC_ALL, '')
import numpy as np
import pandas as pd

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, TensorBoard
from datetime import datetime
from tqdm import tqdm
import cv2
import invasive

########## Globale Variablen etc. ###################


weights_path = "../weights/vgg16_retrain.h5"
model_path = "../models/vgg16_retrained.h5"
img_height = 300
img_width = 400



def make_model(weights_path,img_height,img_width):
    top_weights_path = "../weights/vgg16_top_weights.h5"
    vgg16 = applications.VGG16(
            include_top = False,
            weights = None,
            input_shape = (img_width, img_height, 3))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:],name = 'Flatten_layer'))
    top_model.add(Dense(256, activation = 'relu', name = 'Dense_1'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1,activation = 'sigmoid', name = 'Classifier_layer'))
    model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
    #model.load_weights(weights_path)
    model.compile(loss = 'binary_crossentropy',
                    optimizer=optimizers.Adam(),
                    metrics=['accuracy'])

    return model


model = make_model(weights_path,img_height,img_width)
model = invasive.train_model(model,weights_path,img_width,img_height)
model.save_weights(weights_path)
model.save(model_path)
predictions = invasive.makepredictions(model)
np.save(open('../predictions/vgg16_retrain_predictions.npy', 'w'), predictions)
