import locale

# Set to users preferred locale:
locale.setlocale(locale.LC_ALL, '')
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dropout, GlobalAveragePooling2D, Dense
from keras.callbacks import EarlyStopping
from sklearn import model_selection
from keras import optimizers
#######################Generelle Variablen###################
epochs = 50
batch_size = 32
weights_path = "../weights/vgg16_top_weights.h5"
########### Features Laden################
data= np.load(open('../features/vgg16_features_inorder.npy'))
###########Arrays bauen##########################

labels_csv = pd.read_csv('../data/train_labels.csv')
labels = np.array(labels_csv.invasive.values[0:2295])

print("Data shape: ", data.shape)
print("Labels shape: ", labels.shape)
split_at = 2295/5


train_data = data[split_at:]
test_data = data[:split_at]
train_labels = labels[split_at:]
test_labels = labels[:split_at]

print("train_data shape: ",train_data.shape)
print("train_labels shape: ",train_labels.shape)
########################################

################Top_model bauen###########################

model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:],name = 'GlobalAveragePooling2D_layer'))
model.add(Dense(256, activation = 'relu', name = 'Dense_1'))
model.add(Dropout(0.5))
model.add(Dense(1,activation = 'sigmoid', name = 'Classifier_layer'))

############TODO Optimizer konfigurieren#####################
model.load_weights(weights_path)
model.compile(optimizer=optimizers.Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor="val_loss", patience = 3, verbose = 1, mode= "auto")
model.fit(train_data, train_labels,
        epochs = epochs,
        batch_size = batch_size,
        verbose = 1,
        callbacks = [early_stopping],
        validation_data=(test_data, test_labels))

model.save_weights(weights_path)
