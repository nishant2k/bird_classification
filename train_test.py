# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:22:09 2020

@author: 91811
"""

import numpy as np
import os
import cv2
path1 = "E:/bird3"

def path_f(input):
    return (input+".png")
names = os.listdir("E:/bird3") #names of birds
for i in range(len(names)):
    path2 = path1 + "/" + names[i] #path of each bird directory
    bird_names = os.listdir(path2) # files of each bird 
    for j in range(len(bird_names)):
        
        if (len(bird_names) >= 5):
            choice = np.random.choice(bird_names, int(len(bird_names)//1.25) , replace=False)
            for l in range(len(choice)):
                image = cv2.imread(path2 + "/" + choice[l]  )
                cv2.imwrite("E:/bird/train/" + str(names[i]) + "_" + str(choice[l]) , image)
            
            for k in range(len(choice)):
                bird_names.remove(choice[k])
                
            for m in range(len(bird_names)):
                image2 = cv2.imread(path2 + "/" + bird_names[m] )
                cv2.imwrite("E:/bird/test/" + str(names[i]) + "_" + str(bird_names[m]) , image2)
        else:
            for n in range(len(bird_names)):
                image3 = cv2.imread(path2 +"/" + bird_names[n])
                cv2.imwrite("E:bird/train/" + str(names[i]) + "_" + str(bird_names[n]) , image3)
    
    
    
