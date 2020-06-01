"""Importing the necessery libraries"""
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

"""Function for reaing the images"""
def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

path = "/content/bird_f" # path of bird directory conatining test and train datasets
#train_path = os.path.join(path, "train/*")
test_path = os.path.join(path, "test/*") # test directory path
labels_path2 = os.path.join(path, "test_labels.csv") #train labels path that conatins the unique id of bird with respective breed of the bird


labels_df2 = pd.read_csv(labels_path2)
##Save the breed2id and id2breed dictionary from model.py
model = tf.keras.models.load_model("model3.h5")


l1 = labels_df2["id"] # list of the unique id 
l2 = labels_df2["breed"] # list of corresponding breed of bird

zip1 = list(zip(l1, l2)) #creating zip of id with breed of bird

random.shuffle((zip1)) # randomly shuffling the zip

res1 = list(zip(*zip1))

l1 = list(res1[0]) #extracting the shuffled id's
l2 = list(res1[1]) #extracting the respective breed of l1 list

predicted=[] # creating list for appending the predicetd bird
for i in range(len(l1)):
    image = read_image(path + "/" + "test/" + str(l1[i]) + ".png", 224) #reading the images
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)[0] #predicting the bird index form our model
    label_idx = np.argmax(pred)
    breed_name = id2breed[label_idx] # converting the index to breed with id2breed dictionary
    predicted.append(breed_name)
    
print(accuracy_score(l2,predicted))#calculating the accuracy of our model
