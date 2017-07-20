import locale

# Set to users preferred locale:
locale.setlocale(locale.LC_ALL, '')
import numpy as np
import pandas as pd

import cv2
from tqdm import tqdm
from keras.models import Sequential
from keras.preprocessing.image  import ImageDataGenerator
from keras import applications


#################Variablen Definition#################

train_dir = '../data/train/'
validation_dir = '../data/validation/'
img_height = 300
img_width = 400
batch_size = 32
################ img und label einlesen ###################
train_labels = pd.read_csv('../data/train_labels.csv')
def read_img(img_path,img_width,img_height):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(img_height,img_width))
    return img

train_img = []
for img_path in tqdm(train_labels['name'].iloc[: ]):
    train_img.append(read_img('../data/train/' + str(img_path) +'.jpg',img_width,img_height))

train_img = np.array(train_img, np.float32) / 255
print(train_img.shape)
labels = np.array(train_labels.invasive.values[0:2295])

model = applications.VGG16(
        include_top = False,
        weights = 'imagenet',
        input_shape = (img_width, img_height, 3))
model.summary()

features = model.predict(train_img, batch_size=1, verbose = 1)
np.save(open('../features/vgg16_features_inorder.npy', 'w'), features)

