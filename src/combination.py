import locale

# Set to users preferred locale:
locale.setlocale(locale.LC_ALL, '')
import numpy as np
import pandas as pd

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dropout, Dense
from keras.callbacks import EarlyStopping, TensorBoard
from datetime import datetime
from tqdm import tqdm
import cv2
import invasive
import os.path

def predict_features(path_to_model, name):
    if os.path.isfile('../features/'+name+'.npy'):
        return np.load('../features/'+name+'.npy')
    else:
        model = load_model(path_to_model)
        img_width =400
        img_height=300
        data = []
        for i in xrange(2295):
                data.append(invasive.read_img('../data/train/' + str(i+1) +'.jpg',img_width,img_height))
        data = np.array(data, np.float32)/255
        predictions = model.predict(data, batch_size = 1,
                        verbose = 1)
        np.save(open('../features/'+name+'.npy', 'w'))
        return predictions

def make_model():
    model = Sequential()
    model.add(Dense(1,input_shape = (2,), activation = 'sigmoid'))
    model.compile(optimizer = optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def train_model(model):
    vgg_path = '../models/vgg16_finetuned.h5'
    groundup_path = '../models/groundup.h5'
    feature_vgg = predict_features(vgg_path,"vgg_total")
    feature_groundup = predict_features(vgg_path,"groundup_total")
    data = np.column_stack((feature_vgg,feature_groundup))
    feature = predict_features("../models/resnet50_retrained.h5","resnet50_retrained")
    data = np.column_stack((data,feature))
    feature = predict_features("../models/resnet50_untraineded.h5","resnet50_untrained")
    data = np.column_stack((data,feature))
    feature = predict_features("../models/vgg16_retrained.h5","vgg16_retrained")
    data = np.column_stack((data,feature))
    feature = predict_features("../models/vgg16_untraineded.h5","vgg16_untrained")
    data = np.column_stack((data,feature))
    feature = predict_features("../models/inceptionv3_retrained.h5","inceptionv3_retrained")
    data = np.column_stack((data,feature))
    feature = predict_features("../models/inceptionv3_untraineded.h5","inceptionv3_untrained")
    data = np.column_stack((data,feature))
    train_labels = pd.read_csv('../data/train_labels.csv')
    labels = np.array(train_labels.invasive.values[0:2295])
    split_at = 2295/5
    train_labels = labels[split_at:]
    test_labels = labels[:split_at]
    data = np.column_stack((feature_vgg,feature_groundup))
    train_data = data[split_at:]
    test_data = data[:split_at]
    model.fit(train_data, train_labels,
        epochs = 500,
        batch_size = 32,
        verbose = 1,
        validation_data=(test_data, test_labels))
    return model

model = make_model()
model = train_model(model)
model.save('../models/combination.h5')
