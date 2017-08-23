#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "dnn_conv.h"
#include "demod.h"
#include "global_var.h"

#define PARAM_DATA "neural_net_parameters.bin"
#define DEC_ENV "dec_env.bin"
#define WEIGHTS "weights.bin"
#define BIASES "biases.bin"
#define SIZE_IN "size_in.bin"
#define SIZE_OUT "size_out.bin"
#define KERNEL_X "kernel_x.bin"
#define KERNEL_Y "kernel_y.bin"

#define MALLOC(TYPE, COUNT) ((TYPE *) malloc(sizeof(TYPE) * COUNT))
#define CALLOC(TYPE, COUNT) ((TYPE *) calloc(COUNT, sizeof(TYPE)))

int main(int argc, char* argv[]) {
	// Allocating memory
	/*
	float *inputBuf = MALLOC(float, IN_R * IN_C);
	int *size_in = MALLOC(int, num_layers);
	int *size_out = MALLOC(int, num_layers);
	int *kernel_x = MALLOC(int, num_layers);
	int *kernel_y = MALLOC(int, num_layers);
	*/
	
	struct CNN param;

	float inputBuf[IN_R * IN_C];

	for (int i = 0; i < NUM_LAYERS; i++) {
		for (int j = 0; j < 5; j++) {
			for (int k = 0; k < IN_R * IN_C; k++) {
				param.output[i][j][k] = 0;
			}
		}
	}

	if (!inputBuf) { perror("Could not allocate envelope :("); exit(EXIT_FAILURE); }

	// Opening files
	FILE *fenv = fopen(DEC_ENV, "rb");
	FILE *fin = fopen(SIZE_IN, "rb");
	FILE *fout = fopen(SIZE_OUT, "rb");
	FILE *fx = fopen(KERNEL_X, "rb");
	FILE *fy = fopen(KERNEL_Y, "rb");

	if (!fenv) { perror("Could not open envelope :("); exit(EXIT_FAILURE); } 
	if (!fin) { perror("Could not open size_in :("); exit(EXIT_FAILURE); } 
	if (!fout) { perror("Could not open size_out :("); exit(EXIT_FAILURE); } 
	if (!fx) { perror("Could not open kernel_x :("); exit(EXIT_FAILURE); } 
	if (!fy) { perror("Could not open kernel_y :("); exit(EXIT_FAILURE); } 

	fread(inputBuf, sizeof(float), IN_R * IN_C, fenv);
	fread(param.size_in, sizeof(int), NUM_LAYERS, fin);
	fread(param.size_out, sizeof(int), NUM_LAYERS, fout);
	fread(param.kernel_x, sizeof(int), NUM_LAYERS, fx);
	fread(param.kernel_y, sizeof(int), NUM_LAYERS, fy);

	for (i = 0; i < NUM_LAYERS; i++) {
		printf("size_in[%i] = %i\n", i, param.size_in[i]);
	}

	for (i = 0; i < NUM_LAYERS; i++) {
		printf("size_out[%i] = %i\n", i, param.size_out[i]);
	}

	fclose(fenv);
	fclose(fin);
	fclose(fout);
	fclose(fx);
	fclose(fy);

	// Opening weight and bias binary files
	FILE *fw = fopen(WEIGHTS, "rb");
	FILE *fb = fopen(BIASES, "rb");

	if (!fw) { perror("Could not open weights :("); exit(EXIT_FAILURE); } 
	if (!fb) { perror("Could not open biases :("); exit(EXIT_FAILURE); } 


	// Allocating memory for weights and biases
	// float ****weights = MALLOC(float***, num_layers);
	// float **biases = MALLOC(float*, num_layers);

	// for (int i = 0; i < num_layers; i++) {
	// 	biases[i] = MALLOC(float, size_out[i]);
	// 	fread(biases[i], sizeof(float), size_out[i], fb);

	// 	weights[i] = MALLOC(float**, size_in[i]);
	// 	for (int j = 0; j < size_in[i]; j++) {

	// 		weights[i][j] = MALLOC(float*, size_out[i]);
	// 		for (int k = 0; k < size_out[i]; k++) {
	// 			weights[i][j][k] = MALLOC(float, kernel_x[i] * kernel_y[i]);
	// 			fread(weights[i][j][k], sizeof(float), kernel_x[i] * kernel_y[i], fw);
	// 		}
	// 	} 
	// }

	for (i = 0; i < NUM_LAYERS; i++) {
		fread(&param.biases[i], sizeof(float), param.size_out[i], fb);
		for (int j = 0; j < param.size_in[i]; j++) {
			for (int k = 0; k < param.size_out[i]; k++) {
				fread(&param.weights[i][j][k], sizeof(float), param.kernel_x[i] * param.kernel_y[i], fw);
			}
		}
	}


	clock_t start = clock(), diff;
	printf("Beginning convolution\n");
	// convolution(&param, inputBuf);
	single_layer(&param, inputBuf, 0);
	diff = clock() - start;
	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Convolution took %d seconds %d milliseconds\n", msec/1000, msec%1000);

	// test(&param, 1);
	printf("Yay! I <3 PumpKim!");
	return 0;

}
