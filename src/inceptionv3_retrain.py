import locale

# Set to users preferred locale:
locale.setlocale(locale.LC_ALL, '')
import numpy as np
import pandas as pd

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dropout, GlobalAveragePooling2D, Dense
from keras.callbacks import EarlyStopping, TensorBoard
from datetime import datetime
from tqdm import tqdm
import cv2
import invasive
import os.path

########## Globale Variablen etc. ###################


weights_path = "../weights/inceptionv3_retrain.h5"
model_path = "../models/inceptionv3_retrained.h5"
img_height = 300
img_width = 400



def make_model(weights_path,img_height,img_width):
    inceptionv3 = applications.inception_v3.InceptionV3(
            include_top = False,
            weights = 'imagenet',
            input_shape = (img_width, img_height, 3))

    top_model = Sequential()
    top_model.add(GlobalAveragePooling2D(input_shape=inceptionv3.output_shape[1:],name = 'GlobalAveragePooling2D_layer'))
    top_model.add(Dense(256, activation = 'relu', name = 'Dense_1'))
    top_model.add(Dropout(0.75))
    top_model.add(Dense(1,activation = 'sigmoid', name = 'Classifier_layer'))
    model = Model(inputs=inceptionv3.input, outputs=top_model(inceptionv3.output))
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
    model.compile(loss = 'binary_crossentropy',
                    optimizer=optimizers.Adam(),
                    metrics=['accuracy'])

    return model


model = make_model(weights_path,img_height,img_width)
model = invasive.train_model(model,weights_path,img_width,img_height)
model.save_weights(weights_path)
model.save(model_path)
predictions = invasive.makepredictions(model)
np.save(open('../predictions/inceptionv3_retrain_predictions.npy', 'w'), predictions)
