## -*- coding: utf-8 -*-
#"""
#Created on Tue Jun 27 08:58:57 2017
#
#@author: KHua
#"""
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
#
#n = 0
#samples = 5760
#numA = 80
#data = []
#segment = 10
#output = np.zeros((numA, samples))
#envSize = 576
#env = np.zeros((numA, envSize))
#
#
## Reading RF data
#with open("rf-00-A", "rb") as file:
#    while n < samples * numA:
#        bits = file.read(2)
#        if bits == b"":
#            break
#        i = int.from_bytes(bits, byteorder="little", signed=False)
#        data.append(i)
#        n += 1
#
## Reading log compression table
#file = 'LUT.xlsx'
#x1 = pd.ExcelFile(file)
#lut = x1.parse('Sheet1')
#
## Calculating output, two's complement
#for i in range (0, numA):
#    temp = data[i*samples : (i+1)*samples]
#    for j in range (0, samples):
#        if temp[j] > math.pow(2, 11):
#            output[i, j] = abs(temp[j] - math.pow(2, 12))
#        else:
#            output[i, j] = temp[j]
#
## Calculate the envelope by segmentation
#for i in range (0, numA):
#    temp = output[i]
#    for j in range (0, envSize):
#        data_segment = temp[j*segment : (j+1)*segment]
#        env[i, j] = np.mean(data_segment)
#
## Use LUT for log compression     
#for i in range(0, numA):
#    for j in range (0, envSize):
#        env[i,j] = lut.loc[int(env[i,j])]
        
# for i in range(0, 576):
#     print(env[79, i], end=" ")

with open("prob_map.txt") as f:
    values = np.array([float(line.strip()) for line in f])
values = values.reshape((48, 40))
        
        
    

    
# Show image
fig = plt.figure()
fig = plt.imshow(values, cmap='gray', aspect='auto')        
# plt.plot(range(int(samples/segment)), env[1])