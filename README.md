# CNN

HLS
- Contains files for Vivado HLS tool
- This is the main folder has the best optimization of that time
- Includes main.c and conv.c, with appropiate binary files for weights, biases, kernels, etc.

v1.0
- Contains files for gcc compiler / CPU
- No HLS optimzations
- Format of this code is broken up into several sub-functions for convolution

cnn_code
- "Gold standard"
- Used to compare with outputs of HLS code
- Otherwise, this code is obselete.
- Format of this code is broken up into several sub-functions for convolution
