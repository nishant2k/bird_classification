# In this script we will be doing the audio cutting part of our model
#importing necessery libraries
import pandas as pd
import uuid
import subprocess
from random import randint
import os
from pydub import AudioSegment 
#All the files
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
for i in range(len(b1)): # reading the finally.csv file of each bird breated in make_file.py
  a = pd.read_csv("/content/drive/My Drive/project/bird"+"/"+b1[i]+"/"+"finally.csv")
  for j in range(len(a["Begin Path"])): # Taking length of csv file
    audio1=a["Begin Path"][j][23:58]
    audio2=a["Begin Path"][j][59:] 
    start = a["Begin Time (s)"].values[j]  #starttime
    end = a["End Time (s)"].values[j]   #endtime  
    path = "/content/drive/My Drive/winter2" +"/"+ audio1+ "/" +audio2
    final = "/content/drive/My Drive/project/bird3"+"/"+b1[i]+"/"+"audio_files"+"/"+str(j)+str('.wav')

    startTime = start*1000
    endTime = end*1000

    song = AudioSegment.from_wav(path)
    extract = song[startTime:endTime]
    extract.export(final, format = "wav") # exporting the csv file to their respective audio_file folder of the bird name
