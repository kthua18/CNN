# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:24:59 2017

@author: KHua
"""
import struct
import matplotlib.pyplot as plt
import numpy as np

def show(img):
    plt.imshow(img, cmap='gray', aspect='auto')  

n = 0
buffer = []
with open("output0.bin", "rb") as file:
    bin_val = file.read(4)
    while n < 5 * 48 * 40:
        val = struct.unpack('<f', bin_val)
        buffer.append(val)
        n += 1
        bin_val = file.read(4)
        
countc = 0
out0 = []
for lyr in range(0, 5):
    new1 = []
    for i in range(0, 48):
        new2 = []
        for j in range(0, 40): 
            new2.append(buffer[countc][0])
            countc += 1
        new1.append(new2)
    out0.append(new1)
    
plt.figure(1)
for i in range(0, 5):
    plt.subplot(int("51" + "%s"%(i+1)))
    show(out0[i])
    plt.axis('off')