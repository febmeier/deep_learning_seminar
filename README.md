# deep_learning_seminar
## Kaggle invasive Species classification competition

This code is a simple start for the Kaggle invasive Species classification competition.
(https://www.kaggle.com/c/invasive-species-monitoring)


### Prerequisites
To run the code you will require the following python packages:
```
Keras
Tensorflow (or Theano)
opencv
pandas
numpy
```
First setup the directory architecture:
For this you will need the following directories 1 level above the src directory:
```
data/
models/
features/
predictions/
weights/
```
if you also want to use TensorBoard logs you need to create a directory
```
logs/
```
Next you will need to download the data from
https://www.kaggle.com/c/invasive-species-monitoring/data
or via the commands
```
wget https://www.kaggle.com/c/invasive-species-monitoring/download/test.7z
wget https://www.kaggle.com/c/invasive-species-monitoring/download/train.7z
wget https://www.kaggle.com/c/invasive-species-monitoring/download/train_labels.csv.zip
wget https://www.kaggle.com/c/invasive-species-monitoring/download/sample_submission.csv.zip
```
and unpack all of them into the data directory.

The selfbuilt neural network from groundup.py can now immediately be used by changing into the src directory and running the command
```
python groundup.py
```
This will achieve validation accuracies of up to 94 percent.

To run the pretrained neural networks one must first run the corresponding feature extraction and top_model training.
For example the vgg16 pretrained method can be initialised by running the following python scripts in order:
```
python vgg16_feature_extraction.py
python vgg16_top_trainer.py
```
Afterwards the finetuning script for the vgg16 pretrained model can be executed with the command
```
python vgg16_finetuner.py
```
This will create validation accuracies of up to 97 percent and reach a Kaggle score of over 0.95.

The code uses the vgg16 model ( https://arxiv.org/abs/1409.1556 ) and resnet50 model ( https://arxiv.org/abs/1512.03385 )(currently not properly implemented in our code)
Most of the code is selfwritten, but heavily inspired by the Keras Blog.
(https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
Some code snippits such as the read_img function were copied from kaggle kernels such as

https://www.kaggle.com/finlay/naive-bagging-cnn-pb0-985

and

https://www.kaggle.com/fujisan/use-keras-pre-trained-vgg16-acc-98



###Update:
```
Retrain models included - Not included into predicter yet
GlobalAveragePooling2D added instead of Flatten - Untested
Added inceptionV3 + train untrained models
predicter + combination updated to include retrain + untrained for resnet,vgg,inception in 1  

All retrain  and untrained except inceptionv3 retrain did perform very poorly (<70%)
Changed predicter to predict from groundup/vggfinetune/inceptionv3retrain

```
###Todo:
```
finetune resnet50
update predicter
better use of optimizers by using callbacks reduce LR on plateau etc. optimize use of LR Siehe: https://www.kaggle.com/algila/inception-v3-and-k-fold-in-python-0-98996
Testing - run all retrain/untrained and write down accuracies
Achieve 98%++ accuracy - should be easily possible
Make code more accessible
include setup shell command or python script
Transfer onto other classification tasks
Make Code immediately applicable to all Image classification tasks through input parameters maybe setup.py maybe other ways
```
