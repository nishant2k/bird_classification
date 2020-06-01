"""Importing necessery libraries"""
import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
import random
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

"""Function for building our model"""
def build_model(size, num_classes):
    inputs = Input((size, size, 3))
    backbone = MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet") #Using mobilenetV2 for our model
    backbone.trainable = True
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    


    model = tf.keras.Model(inputs, x)
    return model

"""Function for reading the images"""
def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def parse_data(x, y):
    x = x.decode()

    num_class = 211
    size = 224

    image = read_image(x, size)
    label = [0] * num_class
    label[y] = 1
    label = np.array(label)
    label = label.astype(np.int32)

    return image, label

def tf_parse(x, y):
    x, y = tf.numpy_function(parse_data, [x, y], [tf.float32, tf.int32])
    x.set_shape((224, 224, 3))
    y.set_shape((211))
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


path = "/content/bird_f" # path to the directory conatining the train and test datasets
train_path = os.path.join(path, "train/*") #path of train dataset
#test_path = os.path.join(path, "test/*")
labels_path = os.path.join(path, "train_labels.csv") # path for the labels of train data conating images with their respective bird name


labels_df = pd.read_csv(labels_path)
breed = labels_df["breed"].unique() # creating a breed named list that will be conating the unique birds
print("Number of Breed: ", len(breed)) 

breed2id = {name: i for i, name in enumerate(breed)} # Creating a Dictionary that will conatin breed name with respective make up id
id2breed = {i: name for i, name in enumerate(breed)} # reverse of breed2id


ids = glob(train_path) # conating the train path randomly
labels = []

for image_id in ids:
    image_id = image_id.split("/")[-1].split(".")[0]
    breed_name = list(labels_df[labels_df.id == image_id]["breed"])[0] # breed name will be name of the bird
    breed_idx = breed2id[breed_name] #converting breed name to index
    labels.append(breed_idx) # appending the breed index to labels list

train_x, valid_x = train_test_split(ids, test_size=0.2, random_state=42) # spliting the data for validation
train_y, valid_y = train_test_split(labels, test_size=0.2, random_state=42)


size = 224
num_classes = 211 # numbers of birds
lr = 1e-5 # learning rate of our model
batch = 16
epochs = 20

    ## Model
#Creating the model with function build above
model = build_model(size, num_classes)
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr), metrics=["acc"])
    # model.summary()

    ## Dataset
    #Building the train and vaalid dataset
train_dataset = tf_dataset(train_x, train_y, batch=batch)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    ## Training of the model
callbacks = [
        ModelCheckpoint("model.h5", verbose=1, save_best_only=True), 
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6)
        ] # saving our model as model.h5
train_steps = (len(train_x)//batch) + 1
valid_steps = (len(valid_x)//batch) + 1
#training the model
model.fit(train_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks)
