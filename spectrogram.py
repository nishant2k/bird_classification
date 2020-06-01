# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:52:14 2020

@author: 91811
"""
##
#import matplotlib.pyplot as plt
#import os
##import cv2
#from scipy import signal
#from scipy.io import wavfile
##
#import scipy.io.wavfile as wav
#from numpy.lib import stride_tricks
#import pandas as pd
#import os
#import os
import pandas as pd
#from pydub import AudioSegment 
import numpy as np
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

path = "D:/winter2/Selections" 

a1 = pd.read_csv("D:/project2/csv_files/1.csv") # Reading 1st file from the csv files created in make_file.py script

b1 = list(set(a1["Notes"])) #created a set of all species,no repetition

"""Taking the b1 list as same as in make_file.py"""
for i in range(1,len(files)):
    df = pd.read_csv("D:/project2/csv_files"+"/"+files[i]+".csv")
    df["Diff"] = df["End Time (s)"] - df["Begin Time (s)"]
    b = np.array(df["Notes"])
    for j in range(len(b)):
        if str(b[j]) not in b1:
            b1.append(b[j])
                  
arr = []#Creating array of birds                  
for i in range(len(b1)):
    arr.append(b1[i]) 
arr = np.array(arr)


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, save_f, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    #if plotpath:
    plt.savefig(save_f, bbox_inches="tight")
    #else:
    plt.show()

   # plt.clf()

    return ims

"""Creating spectrograms for each bird audio_files"""
for i in range(len(arr)):
    df = pd.read_csv("D:/project2/bird/"+str(arr[i])+"/"+"finally.csv") #Reading the finnaly.scv file of each bird
    for j in range(len(df["Begin Path"])):
        ims = plotstft("D:/project2/bird/"+str(arr[i])+"/"+"audio_files"+"/"+str(j)+".wav", "D:/project/bird3/"+str(arr[i])+"/"+str(j))
