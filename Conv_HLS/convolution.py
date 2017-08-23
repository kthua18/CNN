# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:06:58 2017

@author: KHua
"""

#def readval(num):
#    return np.fromstring(param.read(num * flt), dtype=np.float32)
        
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import math
import pandas as pd
import struct

n = 0
samples = 5760
numA = 80
data = []
segment = 10
output = np.zeros((numA, samples))
envSize = 576
env = np.zeros((numA, envSize))
convImages = []

inc = 48
inr = 40
factorc = int(envSize / inc); # 576 / 48 = 12
factorr = int(numA / inr); # 80 / 40 = 2

'''
# Reading RF data
with open("rf-00-A", "rb") as file:
    while n < samples * numA:
        bits = file.read(2)
        if bits == b"":
            break
        i = int.from_bytes(bits, byteorder="little", signed=False)
        data.append(i)
        n += 1

# Reading log compression table
file = 'LUT.xlsx'
x1 = pd.ExcelFile(file)
lut = x1.parse('Sheet1')

# Calculating output, two's complement
for i in range (0, numA):
    temp = data[i*samples : (i+1)*samples]
    for j in range (0, samples):
        if temp[j] > math.pow(2, 11):
            output[i, j] = abs(temp[j] - math.pow(2, 12))
        else:
            output[i, j] = temp[j]

# Calculate the envelope by segmentation
for i in range (0, numA):
    temp = output[i]
    for j in range (0, envSize):
        data_segment = temp[j*segment : (j+1)*segment]
        env[i, j] = np.mean(data_segment)

# Use LUT for log compression     
for i in range(0, numA):
    for j in range (0, envSize):
        env[i,j] = lut.loc[int(env[i,j])]
        
#for i in range(0, 576):
#    print(env[79, i], end=" ")

'''

# Decimation of the envelope
# Image of 576x80 to 48x40

dec_env = np.zeros((inr, inc))
temp = np.zeros((factorr, factorc))
for i in range(0, inr):
    for j in range (0, inc):
        temp = env[i * factorr : i * factorr + factorr, 
                   j * factorc : j * factorc + factorc]
        dec_env[i][j] = temp.mean()

# Show image
# plt.imshow(dec_env, cmap='gray', aspect='auto')        
# plt.plot(range(int(samples/segment)), env[1])

# Begin convolution

# Initializing variables
numLayers = 8
imgX = 40
imgY = 48
kernel = [5, 7, 9, 11, 13, 15, 17, 17]
sizeIn = [1, 5, 5, 5, 5, 5, 5, 5]
sizeOut = [5, 5, 5, 5, 5, 5, 5, 1]

biases = [] # layers, sizeout
weightsRev = []
weights = [] # layers, sizein, sizeout, x, y
dec_env = []

convImages = np.zeros([5, 48, 40])
final = np.zeros([48, 40])
test = []
ctest = []

# Reading values generated from C code
with open("biases.txt") as fb:
    biasVals = np.array([float(line.strip()) for line in fb])
    
with open("output.txt") as fc:
    outVals = np.array([float(line.strip()) for line in fc])
 
weights_buf = []
n = 0
with open("weights1.bin", "rb") as file:
    bin_val = file.read(4)
    while n < 24920:
        val = struct.unpack('<f', bin_val)
        weights_buf.append(val)
        n += 1
        bin_val = file.read(4)
        
n = 0
dec_env_buf = []
with open("dec_env.bin", "rb") as file:
    bin_val = file.read(4)
    while n < 48 * 40:
        val = struct.unpack('<f', bin_val)
        dec_env_buf.append(val)
        n += 1
        bin_val = file.read(4)

# Storing the read values in variables 
# Biases
countb = 0
for i in range (0, numLayers):
    new = []
    for j in range(0, sizeOut[i]):
        new.append(biasVals[countb])
        countb = countb + 1
    biases.append(new)
   
# Weights
countw = 0
for i in range (0, numLayers):
    new1 = []
    for j in range(0, sizeIn[i]):
        new2 = []
        for k in range(0, sizeOut[i]):
            new3 = []
            for l in range(0, kernel[i]):
                new4 = []
                for m in range(0, kernel[i]):
                    new4.append(weights_buf[countw][0])
                    countw = countw + 1
                new3.append(new4)
            new2.append(new3)
        new1.append(new2)
    weightsRev.append(new1)

# Decimated envelope
counte = 0;
for i in range(0, 48):
    new = []
    for j in range(0, 40): 
        new.append(dec_env_buf[counte][0])
        counte += 1
    dec_env.append(new)
   
# C Code testing
countc = 0;
for lyr in range(0,5):
    new1 = []
    for i in range(0, 48):
        new2 = []
        for j in range(0, 40): 
            new2.append(outVals[countc])
            countc += 1
        new1.append(new2)
    ctest.append(new1)
 
# Flipping the kernel values
for i in range(0, numLayers):
    new1 = []
    for j in range(0, sizeIn[i]):
        new2 = []
        for k in range(0, sizeOut[i]):
            new3 = []
            for l in range(0, kernel[i]):
                new3.append(list(reversed(weightsRev[i][j][k][l])))
            new2.append(new3)
        new1.append(new2)
    weights.append(new1)
    
#for i in range(0, numLayers):
#    new1 = []
#    for j in range(0, sizeIn[i]):
#        new2 = []
#        for k in range(0, sizeOut[i]):
#            new2.append(list(reversed(weightsRev[i][j][k])))
#        new1.append(new2)
#    weights.append(new1)

    
# Define functions to be used in convolution
def relu(imgIn):
#    imgOut = imgIn.copy()
#    imgOut[imgOut < 0.0] = 0.0
#    return imgOut
    return np.where(imgIn > 0, imgIn, 0)

def sigmoid(imgIn):
    return 1.0 / (1.0 + np.exp(-imgIn))

def convImg(lyr, img, imgIn):
    output = np.zeros_like(imgIn)
    for kernel in weightsRev[lyr][img]:
        output += scipy.signal.correlate2d(imgIn, kernel, mode="same", 
                                          boundary='fill', fillvalue=0)
    return output

layerOutputs = [] # Intermediate output values from each layer
   
# Convolution
# For layer 0 only
for i in range(0, sizeOut[0]):
#    convImages[i] = scipy.signal.convolve2d(dec_env, weights[0][0][i], 
#                    mode="same", boundary='fill', fillvalue=0)
    convImages[i] = scipy.signal.correlate2d(dec_env, weightsRev[0][0][i], 
                    mode="same", boundary='fill', fillvalue=0)
    convImages[i] += biases[0][i]

convImages = relu(convImages)
layerOutputs.append(convImages.copy())

# For layers 1-6    
for i in range(1, numLayers - 1):
    for j in range(0, sizeOut[i]):
        convImages[j] = convImg(i, j, convImages[j])
        convImages[j] += biases[i][j]
    convImages = relu(convImages)
    layerOutputs.append(convImages.copy())

# For layer 7 only (last layer)
for i in range(0, sizeIn[7]):
    convImages[i] = scipy.signal.correlate2d(convImages[i], weightsRev[7][i][0], 
                    mode="same", boundary='fill', fillvalue=0)
    final += convImages[i]
    
final += biases[7][0]
layerOutputs.append(final.copy())
final = sigmoid(final)

# Plotting the images
def show(img):
    plt.imshow(img, cmap='gray', aspect='auto')  

for i in range(0, 8):
    print ("layer = %s" % (i))
    for j in range(0, sizeOut[i]):
        minval = np.min(layerOutputs[i][j])
        maxval = np.max(layerOutputs[i][j])
        print ("img = %s" % (j))
        print ("min = %s, max = %s\n" % (minval, maxval))

for i in range(0, 7):
    plt.figure(i + 1)
    for j in range(0, sizeOut[i]):
        plt.subplot(int("51" + "%s"%(j+1)))
        show(layerOutputs[i][j])
        plt.axis('on')
     
plt.figure(8)
for i in range(0, 5):
    plt.subplot(int("51" + "%s"%(i+1)))
    show(ctest[i])
    plt.axis('on')

    





























