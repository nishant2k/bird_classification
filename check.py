#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:14:24 2020

@author: nishant
"""
import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

path = "/home/nishant/Documents/bird_f"
train_path = os.path.join(path, "train/*")
test_path = os.path.join(path, "test/*")
labels_path = os.path.join(path, "labels.csv")


labels_df = pd.read_csv(labels_path)
breed = labels_df["breed"].unique()
print("Number of Breed: ", len(breed))

breed2id = {name: i for i, name in enumerate(breed)}
id2breed = {i: name for i, name in enumerate(breed)}

model = tf.keras.models.load_model("model.h5")

#for i, path in tqdm(enumerate(valid_x[:10])):
l1 = labels_df["id"][6009:]
l2 = labels_df["breed"][6009:]

zip1 = list(zip(l1, l2))

random.shuffle((zip1))

res1 = list(zip(*zip1))

l1 = list(res1[0])
l2 = list(res1[1]) #actual

predicted=[]
for i in range(len(l1)):
    image = read_image(path + "/" + "test/" + str(l1[i]) + ".png", 224)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)[0]
    label_idx = np.argmax(pred)
    breed_name = id2breed[label_idx]
    predicted.append(breed_name)
    
print(accuracy_score(l2 , predicted))
    


#ori_breed = id2breed[valid_y[i]]
#ori_image = cv2.imread(path, cv2.IMREAD_COLOR)
#image = read_image(path + "/" + "test/INSECT_3_314.png", 224)
#image = np.expand_dims(image, axis=0)
#pred = model.predict(image)[0]
#label_idx = np.argmax(pred)
#breed_name = id2breed[label_idx]
#predicted.append(breed_name)
#
#
#ori_image = cv2.putText(ori_image, breed_name, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#ori_image = cv2.putText(ori_image, ori_breed, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

 #cv2.imshow(f"/home/nishant/Downloads/a.png", ori_image)
