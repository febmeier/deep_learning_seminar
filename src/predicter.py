import locale

# Set to users preferred locale:
locale.setlocale(locale.LC_ALL, '')

import numpy as np
import pandas as pd

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
#TODO ALLL!!!!!!!!!!!!!!!!!!!!!!!

def roundToBinary(prediction):
    predictions_rounded = []
    for pred in prediction:
        if (pred > .5):
            predictions_rounded.append("1")
        else:
            predictions_rounded.append("0")
    return predictions_rounded

def csvpredict(prediction,name):
        sample_submission = pd.read_csv('../data/sample_submission.csv')
        sample_submission['invasive'] = roundToBinary(prediction)
        sample_submission.to_csv('../predictions/'+name, index = None)

def predict_from_all():
        predictions_vgg = np.load(open('../predictions/vgg16_prediction_array.npy'))
        predictions_groundup = np.load(open('../predictions/groundup_prediction_array.npy'))
        data = np.column_stack((predictions_vgg,predictions_groundup))
        model = load_model('../models/combination.h5')
        predictions = model.predict(data, batch_size = 1,verbose = 1)
        return predictions

path_vgg = 'submit_vgg.csv'
path_groundup ='submit_groundup.csv'
predict_path = 'submit_1337.csv'
predictions_vgg = np.load(open('../predictions/vgg16_prediction_array.npy'))
predictions_groundup = np.load(open('../predictions/groundup_prediction_array.npy'))
csvpredict(predictions_vgg, path_vgg)
csvpredict(predictions_groundup, path_groundup)
csvpredict(predict_from_all(), predict_path)
