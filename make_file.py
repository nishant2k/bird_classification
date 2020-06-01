# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:00:57 2020

@author: 91811
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:27:23 2020

@author: 91811
"""

import pandas as pd
import os

files = ["1", "P03929TEA1_20170505_113500.Table.1.selections", "P03929TEA1_20170505_190500.Table.1.selections", "P03929TEA1_20170506_070001.Table.1.selections",
         "P03929TEA1_20170506_143500.Table.1.selections", "P03929TEA1_20170506_193500.Table.1.selections", "P03929TEA1_20170507_072002.Table.1.selections",
         "P03929TEA1_20170507_120500.Table.1.selections", "P03929TEA1_20170507_203500.Table.1.selections", "P03929TEA1_20170508_062900.Table.1.selections",
         "P03929TEA1_20170508_120400.Table.1.selections", "P03929TEA1_20170508_190400.Table.1.selections", "P03929TEA1_20170509_061900.Table.1.selections", "P03929TEA1_20170509_123400.Table.1.selections",
         "P03929TEA1_20170509_200400.Table.1.selections", "P03929TEA1_20170510_060400.Table.1.selections", "P03929TEA1_20170510_143400.Table.1.selections", "P03929TEA1_20170510_203400.Table.1.selections",
         "P03929TEA1_20170511_064801.Table.1.selections", "P03929TEA1_20170511_143300.Table.1.selections", "P03929TEA1_20170511_190300.Table.1.selections", "P03929TEA1_20170512_064801.Table.1.selections",
         "P03932FOR1_20170505_064101.Table.1.selections", "P03932FOR1_20170505_110600.Table.1.selections", "P03932FOR1_20170505_200600.Table.1.selections", "P03932FOR1_20170508_060500.Table.1.selections",
         "P03932FOR1_20170508_130500.Table.1.selections", "P03932FOR1_20170508_193500.Table.1.selections", "P03932FOR1_20170511_075403.Table.1.selections", "P03932FOR1_20170511_143400.Table.1.selections", 
         "P03932FOR1_20170511_193400.Table.1.selections", "P03932FOR1_20170512_070302.Table.1.selections", "P03939FOR2_20170505_072502.Table.1.selections", "P03939FOR2_20170505_130500.Table.1.selections",
         "P03939FOR2_20170505_190500.Table.1.selections", "P03939FOR2_20170506_075503.Table.1.selections", "P03939FOR2_20170506_133500.Table.1.selections", "P03939FOR2_20170506_200500.Table.1.selections",
         "P03939FOR2_20170507_070502.Table.1.selections", "P03939FOR2_20170507_143500.Table.1.selections", "P03939FOR2_20170507_200500.Table.1.selections", "P03939FOR2_20170508_072902.Table.1.selections",
         "P03939FOR2_20170508_113400.Table.1.selections", "P03939FOR2_20170508_200400.Table.1.selections", "P03939FOR2_20170509_062400.Table.1.selections", "P03939FOR2_20170509_133400.Table.1.selections",
         "P03939FOR2_20170509_193400.Table.1.selections", "P03939FOR2_20170510_061400.Table.1.selections", "P03939FOR2_20170510_120400.Table.1.selections", "P03939FOR2_20170510_190400.Table.1.selections",
         "P03939FOR2_20170511_064301.Table.1.selections", "P03939FOR2_20170511_133300.Table.1.selections", "P03939FOR2_20170511_193300.Table.1.selections", "P03939FOR2_20170512_072802.Table.1.selections",
         "P03939FOR2_20170512_130300.Table.1.selections", "P03940TEA2_20170505_061500.Table.1.selections", "P03940TEA2_20170505_133500.Table.1.selections", "P03940TEA2_20170505_193500.Table.1.selections" 
         ]
##extraxt filed as csv
for i in range(len(files)):
  a=pd.read_csv(path+"/"+files[i]+".txt", sep="\t")
  #print(a["Begin Time (s)"])
##converting txt files to csv files and saving them
  a.to_csv("/content/csv_file"+"/"+files[i]+".csv")
  
#Reading any one file
a1 = pd.read_csv("C:/Users/91811/Downloads/Selections-20200514T132501Z-001/Selections2/1.csv")
a1["Diff"] = a1["End Time (s)"] - a1["Begin Time (s)"] # create new column for differerbce

b1 = list(set(a1["Notes"])) #created a set of all species,no repetition

# define the access rights
access_rights = 0o755

path1 = "C:/Users/91811/Downloads/Selections-20200514T132501Z-001/S3/bird"

#make bird directory
try:
    os.mkdir(path1,access_rights)    
except FileExistsError:
    print("Directory already exists")


#making directories with bird names for a file
for i in range(len(b1)):
    path2=path1+"/"+str(b1[i])
    try:
        os.mkdir(path2,access_rights)    
    except FileExistsError:
        print("Directory already exists")
        continue

#grouping the dataframe according to the bird name code
c=a1.groupby("Notes")
for i in range(len(b1)):
    d=c.get_group(b1[i])
    path2=path1+"/"+str(b1[i])+"/finally.csv"
    e=d.to_csv(path2)  #exporting the final csv into their respective folders. The csv will be conatining the dataset of each bird 
                          #of the first file

         
         
 ## Doing same for the rest of the files
for i in range(1,len(files)):
  df = pd.read_csv("C:/Users/91811/Downloads/Selections-20200514T132501Z-001/Selections2"+"/"+files[i]+".csv")
  df["Diff"] = df["End Time (s)"] - df["Begin Time (s)"]
  b = df["Notes"]  #Creating list of birds for a paticular file of the loop
  for j in range(len(b)):
    if str(b[j]) not in b1:
      b1.append(b[j])  # Appending the b1 list(containi birds of first file) if the bird name is not in the b1 list
      path3 = path1+"/"+str(b[j]) # making path for creating directories of the new bird
      try:
          os.mkdir(path3,access_rights)    #making the directory of particular bird#
      except FileExistsError:
          print("Directory already exists")
          continue
      c11 = df.groupby("Notes") 
      d22 = c11.get_group(b[j])
      path4 = path3+"/"+"finally.csv"
      e1=d22.to_csv(path4)  #exporting the final csv of the new bird to their respective folder
    else:
      c11 = df.groupby("Notes")
      d22 = c11.get_group(b[j])
      e1 = d22.to_csv("m.csv", header=False)
      with open('m.csv', 'r') as f1, open(path1+"/"+str(b[j])+"/finally.csv", 'a+') as f2:
          f2.write(f1.read()) 
        
