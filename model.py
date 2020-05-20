#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:28:48 2020

@author: nishant
"""

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

def build_model(size, num_classes):
    inputs = Input((size, size, 3))
    backbone = MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet")
    backbone.trainable = True
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(num_classes, activation="softmax")(x)
    
#    x = GlobalAveragePooling2D()(x)
#    x = Dropout(0.2)(x)
#    x = Dense(1024, activation="relu")(x)
#   #x = Dense(num_classes, activation="softmax")(x)
#    
#    
#    x = GlobalAveragePooling2D()(x)
#    x = Dropout(0.2)(x)
#    x = Dense(1024, activation="relu")(x)
#    x=Dense(num_classes, activation="softmax")(x)



    model = tf.keras.Model(inputs, x)
    return model

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


path = "/home/nishant/Documents/bird_f"
train_path = os.path.join(path, "train/*")
test_path = os.path.join(path, "test/*")
labels_path = os.path.join(path, "labels.csv")


labels_df = pd.read_csv(labels_path)
breed = labels_df["breed"].unique()
print("Number of Breed: ", len(breed))

breed2id = {name: i for i, name in enumerate(breed)}

ids = glob(train_path)
labels = []

for image_id in ids:
    image_id = image_id.split("/")[-1].split(".")[0]
    breed_name = list(labels_df[labels_df.id == image_id]["breed"])[0]
    breed_idx = breed2id[breed_name]
    labels.append(breed_idx)


labels_path2 = os.path.join(path, "labels_t.csv")


labels_df2 = pd.read_csv(labels_path2)
breed2 = labels_df2["breed"].unique()
print("Number of Breed: ", len(breed2))

breed2id2 = {name: j for j, name in enumerate(breed2)}

ids2 = glob(test_path)
labels2 = []

for image_id in ids2:
    image_id = image_id.split("/")[-1].split(".")[0]
    breed_name2 = list(labels_df2[labels_df2.id == image_id]["breed"])[0]
    breed_idx2 = breed2id2[breed_name2]
    labels2.append(breed_idx2)

#ids = ids[:1000]
#labels = labels[:1000]

    ## Spliting the dataset
#train_x, valid_x = train_test_split(ids, test_size=0.2, random_state=42)
#train_y, valid_y = train_test_split(labels, test_size=0.2, random_state=42)
train_x = ids
train_y = labels
valid_x = ids2
valid_y = labels2
zip1 =list(zip(train_x,train_y))
zip2 = list(zip(valid_x,valid_y))

random.shuffle(zip1)
random.shuffle(zip2)

res1= list(zip(*zip1))

res2= list(zip(*zip2))

train_x = list(res1[0])
train_y = list(res1[1])
valid_x = list(res2[0])
valid_y = list(res2[1])


    ## Parameters
size = 224
num_classes = 211
lr = 1e-4
batch = 16
epochs = 10

    ## Model
model = build_model(size, num_classes)
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr), metrics=["acc"])
    # model.summary()

    ## Dataset
train_dataset = tf_dataset(train_x, train_y, batch=batch)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    ## Training
callbacks = [
        ModelCheckpoint("model.h5", verbose=1, save_best_only=True),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6)
        ]
train_steps = (len(train_x)//batch) + 1
valid_steps = (len(valid_x)//batch) + 1
model.fit(train_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks)