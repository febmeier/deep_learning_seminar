import locale

# Set to users preferred locale:
locale.setlocale(locale.LC_ALL, '')
import numpy as np
import pandas as pd

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dropout, Dense
from keras.callbacks import EarlyStopping, TensorBoard
from datetime import datetime
from tqdm import tqdm
import cv2


def read_img(img_path,img_width,img_height):
        img = cv2.imread(img_path)
        img = cv2.resize(img,(img_height,img_width))
        return img

def makepredictions(model):
        img_height = 300
        img_width = 400
        data = []
        for i in xrange(1531):
                data.append(read_img('../data/test/' + str(i+1) +'.jpg',img_width,img_height))
        data = np.array(data, np.float32) /255
        predictions = model.predict(data, batch_size = 1,
                    verbose = 1)
        return predictions

def train_model(model, weights_path,img_width,img_height):
        train_labels = pd.read_csv('../data/train_labels.csv')

        data = []
        for img_path in tqdm(train_labels['name'].iloc[: ]):
            data.append(read_img('../data/train/' + str(img_path) +'.jpg',img_width,img_height))

        data = np.array(data, np.float32)
        labels = np.array(train_labels.invasive.values[0:2295])
        split_at = 2295/5
        train_data = data[split_at:]
        test_data = data[:split_at]
        train_labels = labels[split_at:]
        test_labels = labels[:split_at]
        epochs = 50
        batch_size = 32
        datagen = ImageDataGenerator(rescale=1./255,
                            rotation_range=10,
                            width_shift_range=.1,
                            height_shift_range=.1,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=False,
                            fill_mode='reflect')
        validgen = ImageDataGenerator(rescale = 1./255)
        train_gen = datagen.flow(
            train_data,train_labels,
            batch_size = batch_size,
            shuffle = True)
        val_gen = validgen.flow(
            test_data,test_labels,
            batch_size = batch_size,
            shuffle = True)

        train_samples = len(train_labels)
        validation_samples= len(test_labels)
        early_stopping = EarlyStopping(monitor="val_loss", patience = 3, verbose = 1, mode= "auto")
        now = datetime.now()

        # "_tf_logs" is my Tensorboard folder. Change this to your setup if you want to use TB
        logdir = "../logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        tb = TensorBoard(log_dir=logdir)

        model.fit_generator(
                train_gen, epochs = epochs,
                steps_per_epoch = int(train_samples/batch_size),
                validation_data=val_gen,
                validation_steps = int(validation_samples/batch_size),
                verbose = 1,
                callbacks=[early_stopping,tb])
        model.save_weights(weights_path)
        return model
