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

########## Globale Variablen etc. ###################


weights_path = "../weights/vgg16_finetuned_weights.h5"
model_path = "../models/vgg16_finetuned.h5"
img_height = 300
img_width = 400



def make_model(weights_path,img_height,img_width):
    top_weights_path = "../weights/vgg16_top_weights.h5"
    vgg16 = applications.VGG16(
            include_top = False,
            weights = 'imagenet',
            input_shape = (img_width, img_height, 3))

    top_model = Sequential()
    top_model.add(GlobalAveragePooling2D(input_shape=vgg16.output_shape[1:],name = 'GlobalAveragePooling2D_layer'))
    top_model.add(Dense(256, activation = 'relu', name = 'Dense_1'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1,activation = 'sigmoid', name = 'Classifier_layer'))

    top_model.load_weights(top_weights_path)
    model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
    #model.load_weights(weights_path)
    for layer in model.layers[:15]:
        layer.trainable = False


    sgd = optimizers.SGD(lr = 0.0001, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(loss = 'binary_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])

    return model


model = make_model(weights_path,img_height,img_width)
model = invasive.train_model(model,weights_path,img_width,img_height)
model.save_weights(weights_path)
model.save(model_path)
predictions = invasive.makepredictions(model)
np.save(open('../predictions/vgg16_prediction_array.npy', 'w'), predictions)
