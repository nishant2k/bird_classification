# -*- coding: utf-8 -*-
""" Manually solitting the dataset into train_test because our data contains only one file for few birds"""
#Importing necessery libraries
import numpy as np
import os
import cv2
path1 = "E:/bird3"

#def path_f(input):
 #   return (input+".png")
names = os.listdir("E:/bird3") #names of birds in ana list
for i in range(len(names)):
    path2 = path1 + "/" + names[i] #path of each bird directory
    bird_names = os.listdir(path2) # files of each bird 
        
    if (len(bird_names) >= 5): # Manually selecting the birds with more then 4 images 
         choice = np.random.choice(bird_names, int(len(bird_names)//1.25) , replace=False) # Splitting the data randomly into 80:20 ratio
         for l in range(len(choice)):
         image = cv2.imread(path2 + "/" + choice[l]  )
         cv2.imwrite("E:/bird/train/" + str(names[i]) + "_" + str(choice[l]) , image) # Writing the image to train folder
            
         for k in range(len(choice)):
              bird_names.remove(choice[k]) #Removing the choice list 
                
         for m in range(len(bird_names)): # Writing the rest of the images to the test folder
              image2 = cv2.imread(path2 + "/" + bird_names[m] )
              cv2.imwrite("E:/bird/test/" + str(names[i]) + "_" + str(bird_names[m]) , image2)
     else:  #Else expoting the image to the train dataset
         for n in range(len(bird_names)):
              image3 = cv2.imread(path2 +"/" + bird_names[n])
              cv2.imwrite("E:bird/train/" + str(names[i]) + "_" + str(bird_names[n]) , image3)
    
    
    
