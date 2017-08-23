/* 
 * dnn_conv.c
 * Implements convolutional neural network (CNN) to be used in the Zynq 7000x 
 * chip. Input is the image of interest, output is a probability map. Convolution
 * is performed in the spatial domain.
 *
 * @author      Kim Hua
 * @version     1.0, 8 Aug 2017
 *
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "dnn_conv.h"
#include "demod.h"
#include "global_var.h"

#define MALLOC(TYPE, COUNT) ((TYPE *) malloc(sizeof(TYPE) * COUNT))
#define CALLOC(TYPE, COUNT) ((TYPE *) calloc(COUNT, sizeof(TYPE)))
#define PARAM_DATA "neural_net_parameters.bin"

static int num_layers; // number of layers in CNN
static int max_depth = 0; // maximum depth (input images) of all the layers
static int *size_in, *size_out; // the input and output depths of the kernels
static int *kernel_x, *kernel_y; // kernel dimensions for all the layers
static float ****weights, **biases; // matrix of kernel and bias values
float ***output;
float *prob_map;


/*
 * Reorders the kernels from Tensorflow format to the format used in this
 * module. 
 *
 * The Tensorflow format is for every layer with n kernels, values 0 to n 
 * represent the first values of each kernel; values n to 2n represent the 2nd
 * values of each kernel, etc.
 *
 * The format of the weights used in this module is values 0 to n represent
 * one single kernel, the first kernel; values n to 2n represent the 2nd
 * kernel, etc.
 *
 * \param[in] buffer pointer to the original kernel data
 * \param[in] kernel_x horizontal kernel size
 * \param[in] kernel_y vertical kernel size
 * \param[in] total_kernels number of kernels
 */

static void reorder_kernel(const float *buffer,
                           const int layer_idx, // layer
                           const int in, // size in
                           const int out, // size out
                           const int total_kernels, // number of kernels, sizein * sizeout
                           const int kernel_size) { // number of values in kernels
	for (int i = 0; i < in; i++) {
		for (int j = 0; j < out; j++) {
			for (int k = 0; k < kernel_size; k++) { 
				weights[layer_idx][i][j][k] = buffer[k * total_kernels + i * out + j];
			}
		}
	}
}

/*
 * Reads the file; allocates appropiate memory for kernel_x, kernel_y,
 * size_in, size_out, weights, and biases; populates the arrays in the
 * appropiate format.
 * 
 * kernel_x, kernel_y: the x and y dimensions to describe shape of kernel.
 * size_in, size_out: the input and output depth of each layer.
 * weights: 4D array for layers, images, kernels, and values.
 * biases: 2D array for layers, images.
 * output: 3D array of the intermediate outputs of each layer. Index 0
 *         refers to the initial input image.
 * prob_map: The probability map produced at the end of the convolution.
 */
void load(const char *filename) {
	FILE *file = fopen(filename, "rb");

  	if(!file) {
  		perror("File error");
  		exit(EXIT_FAILURE);
  	}

  	float temp;
  	fread(&temp, sizeof(float), 1, file);
  	num_layers = (int) temp;

  	// Allocating memory for layer shape
  	size_in = MALLOC(int, num_layers);
  	size_out = MALLOC(int, num_layers);
  	kernel_x = MALLOC(int, num_layers);
  	kernel_y = MALLOC(int, num_layers);
  	weights = MALLOC(float***, num_layers);
  	biases = MALLOC(float*, num_layers);
	output = CALLOC(float**, num_layers); 
	prob_map = CALLOC(float, IN_R * IN_C);

  	for (int i = 0; i < num_layers; i++) {
  		// Reading the first 4 values and storing 
  		float temp[4];
  		fread(temp, sizeof(float), 4, file);
  		kernel_y[i] = (int) temp[0];
  		kernel_x[i] = (int) temp[1];
  		size_in[i] = (int) temp[2];
  		size_out[i] = (int) temp[3];
  		int num_kernels = size_out[i] * size_in[i];
  		int kernel_size = kernel_x[i] * kernel_y[i];

  		if (size_out[i] > max_depth) {
  			max_depth = size_out[i];
  		}

  		// Allocating memory for weights and biases
  		biases[i] = MALLOC(float, size_out[i]);
  		weights[i] = MALLOC(float**, size_in[i]);
  		output[i] = CALLOC(float*, size_in[i]); 
  		for (int j = 0; j < size_in[i]; j++) {
  			weights[i][j] = MALLOC(float*, size_out[i]);
  			output[i][j] = CALLOC(float, IN_R * IN_C); 
  			for (int k = 0; k < size_out[i]; k++) {
  				weights[i][j][k] = MALLOC(float, kernel_size);
  			}
  		}

  		// Reading and populatings weights
  		float *raw_data = MALLOC(float, num_kernels * kernel_size);
  		fread(raw_data, sizeof(float), num_kernels * kernel_size, file);
  		reorder_kernel(raw_data, i, size_in[i], size_out[i], num_kernels, kernel_size);
  		free(raw_data);

  		// Reading and populating biases
  		raw_data = MALLOC(float, size_out[i]);
  		fread(raw_data, sizeof(float), size_out[i], file);
  		for (int j = 0; j < size_out[i]; j++) {
  			biases[i][j] = raw_data[j]; 
  		}

  		free(raw_data);
  	}

	fclose(file);
}

/* 
 * Returns true if a coordinate is within the bounds of the image, 
 * false otherwise.
 *
 * x, y: the x and y "coordinates" of the image. (0, 0) is the top left corner
 *       of the image. 
 */
bool within_bounds(const int x, const int y) {
	return (x >= 0 && x < IN_C) &&
		   (y >= 0 && y < IN_R);
}

/* 
 * Takes a 578x40 B-mode image single pointer and reduce it to 48x40.
 * Stores the values in the first layer of output. 
 *
 * env: the input signal, or BMode
 */
void decimation(uint8_t env[BMODE_X * BMODE_Y]) {
	int x = BMODE_X / IN_R; // 576 / 48 = 12
	int y = BMODE_Y / IN_C; // 80 / 40 = 2
	int sum = 0;
	static float *buffer;
	buffer = MALLOC(float, IN_R * IN_C);

	int count = 0;

	for (int i = 0; i < IN_C; i++) {
		for (int j = 0; j < IN_R; j++) {
			for (int k = 0; k < y; k++) {
				for (int l = 0; l < x; l++) {
					sum += env[(i * BMODE_X * 2) + (j * x) + (BMODE_X * k) + l];
				}
			}
			float average = sum / (x * y);
			buffer[count] = average;
			count++;
			sum = 0;
		}
	}

	// Transpose matrix
	for (int i = 0; i < IN_R; i++) {
		for (int j = 0; j < IN_C; j++) {
			output[0][0][IN_C * i + j] = buffer[IN_R * j + i];
		}
	}

	// Normalization
	for (int i = 0; i < IN_R * IN_C; i++) {
		output[0][0][i] = output[0][0][i] / 255.0;
	}
}

/* 
 * Performs convolution for 1 window and returns the value.
 *
 * kernel_x, kernel_y: The number of columns and rows of the kernel.
 * center_x, center_y: The center "coordinates" of the image.
 * lyr: The current layer to be calculated.
 * img: The current image to be calculated, or size_in.
 * kernel: The current kernel to be calculated, or size_out.
 */
float conv_kernel(const int kernel_x,
				  const int kernel_y,
				  const int center_x, 
				  const int center_y, 
				  const int lyr,
				  const int img, 
			  	  const int kernel) { 
	int x_coor; 
	int y_coor;
	int k_idx, i_idx;
	float k_val, i_val;
	float sum = 0;
	int pad_zeros = (kernel_x - 1) / 2; // NOTE: Valid only if kernel_x is odd

	for (int y = -pad_zeros; y <= pad_zeros; y++) {
		y_coor = center_y + y; 
		for (int x = -pad_zeros; x <= pad_zeros; x++) {
			x_coor = center_x + x;
			if (!within_bounds(x_coor, y_coor)) {
				continue;
			}
			k_idx = ((y + pad_zeros) * kernel_x) + (x + pad_zeros);
			k_val = weights[lyr][img][kernel][k_idx]; 
			i_idx = IN_C * y_coor + x_coor;
			i_val = output[lyr][img][i_idx]; 
			sum += k_val * i_val;
		}
	}
	return sum; 
}

/* 
 * Performs convolution for 1 image and saves into output. The input for
 * layer n + 1 are the values from layer n. If the final layer is reached,
 * performs convolution and saves into prob_map. 
 * 
 * kernel_x, kernel_y: The number of columns and rows of the kernel.
 * lyr: The current layer to be calculated.
 * img: The current image to be calculated, or size_in.
 * kernel: The current kernel to be calculated, or size_out.
 */
void conv_image(const int kernel_x,
				const int kernel_y,
				const int lyr,
				const int img, 
			  	const int kernel) { 

	int i_idx;

	for (int r = 0; r < IN_R; r++) { 
		for (int c = 0; c < IN_C; c++) { 
			i_idx = IN_C * r + c;
			if (lyr < num_layers - 1) {
				output[lyr + 1][kernel][i_idx] += conv_kernel(kernel_x, kernel_y, 
														 c, r, lyr, img, kernel);

			// If on final layer, save values into prob_map.
			} else {
				prob_map[i_idx] += conv_kernel(kernel_x, kernel_y, c, r, 
											  lyr, img, kernel);
			}
		}
	}
}

/*
 * Performs convolution for 1 layer. 
 *
 * layer: The current layer to be calculated. 
 */
void conv_layer(const int layer) {
	for (int i = 0; i < size_out[layer]; i++) {
		for (int j = 0; j < size_in[layer]; j++) {
			conv_image(kernel_x[layer], kernel_y[layer], layer, j, i);
		}
	}	
}

/*
 * Performs ReLU on one layer, and overwrites the value at that
 * location in the variable output. 
 *
 * layer: The current layer to be calculated.
 */
void relu(const int layer) {
	for (int i = 0; i < size_out[layer]; i++) {
		for (int j = 0; j < IN_R * IN_C; j++) {
			output[layer + 1][i][j] = fmaxf(output[layer + 1][i][j], 0);
			
		}
	}
}

/*
 * Adds the bias of one layer, and overwrites the result at that location
 * in output. If the final layer is reached, adds the bias to prob_map.
 *
 * layer: The current layer to be calculated 
 */
void add_bias(const int layer) {
	if (layer < num_layers - 1) {
		for (int i = 0; i < size_out[layer]; i++) {
			for (int j = 0; j < IN_R * IN_C; j++) {
				output[layer + 1][i][j] += biases[layer][i]; 
			}
		}
	// If on final layer, save values into prob_map.
	} else {
		for (int i = 0; i < IN_R * IN_C; i++) {
			prob_map[i] += biases[layer][0];
		}
	}
}

/*
 * Performs sigmond on the prob_map.
 */
void sigmoid() {
	for (int i = 0; i < IN_R * IN_C; i++) {
		prob_map[i] = 1.0 / (1.0 + expf(-prob_map[i]));
	}
}

/*
 * Performs entire convolution, from B-mode to probability map. Returns
 * a pointer to the probability map.
 */
float* convolution(float input[IN_R * IN_C]) {
	// Start test!
	load(PARAM_DATA);
	for (int i = 0; i < IN_R * IN_C; i++) {
		output[0][0][i] = input[i];
	}
	// End test!

	for (int i = 0; i < num_layers - 1; i++) {
		// Timing each layer of the convolution
		clock_t start = clock(), diff;
		conv_layer(i);
		diff = clock() - start;
		int msec = diff * 1000 / CLOCKS_PER_SEC;
		printf("Conv_layer %d took %d seconds %d milliseconds\n", 
			   i, msec/1000, msec%1000);
		
		add_bias(i);
		relu(i);
	}

	conv_layer(num_layers - 1);
	add_bias(num_layers - 1);
	sigmoid();
	make_files();
	return prob_map;
}

/* Frees the memory and variables that contain the weights, biases, output,
 * prob_map, and the helper variables size_in, size_out, kernel_x, kernel_y.
 *
 * Let it go... let it goooooo
 */
void free_memory() {
	for (int i = 0; i < num_layers; i++) {
		free(biases[i]);
		for (int j = 0; j < size_in[i]; j++) {
			free(output[i][j]);
			for (int k = 0; k < size_out[i]; k++) {
				free(weights[i][j][k]);
			}
		}
	}

	free(size_in);
	free(size_out);
	free(kernel_x);
	free(kernel_y);
	free(prob_map);
}

// Ignore everything below this
void test() {
	for (int i = 0; i < IN_R * IN_C; i++) {
		printf("%f ", output[0][0][i]);
	}
}

void make_files() {
	// Reading the weights and saving into file ---------------------------------------
	FILE *fweights = fopen("weights.bin", "wb");
	if (fweights == NULL) {
		printf("Could not open weights.bin!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < num_layers; i++) {
		for (int j = 0; j < size_in[i]; j++) {
			for (int k = 0; k < size_out[i]; k++) {
				for (int l = 0; l < kernel_x[i] * kernel_y[i]; l++) {
					// fprintf(fweights, "%f\n", weights[i][j][k][l]);
					fwrite(&weights[i][j][k][l], sizeof(float), 1, fweights);
				}
			}
		}
	}
	fclose(fweights);
	printf("Finished weights!\n");

	// Reading the biases and saving into file ---------------------------------------
	FILE *fbiases = fopen("biases.bin", "wb");
	if (fbiases == NULL) {
		printf("Could not open biases.txt!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < num_layers; i++) {
		for (int j = 0; j < size_out[i]; j++) {
			// fprintf(fbiases, "%f\n", biases[i][j]);
			fwrite(&biases[i][j], sizeof(float), 1, fbiases);
		}
	}
	fclose(fbiases);
	printf("Finished biases!\n");

	// Reading the input image and saving into file ---------------------------------------
	FILE *fenv = fopen("dec_env.bin", "wb");
	if (fenv == NULL) {
		printf("Could not dec_env.bin!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < IN_R * IN_C; i++) {
		// fprintf(fenv, "%f\n", output[0][0][i]);
		fwrite(&output[0][0][i], sizeof(float), 1, fenv);
	}
	fclose(fenv);
	printf("Finished dec_env!\n");

	// Reading the 1st intermediate output and saving into file ---------------------------------------
	FILE *fsizein = fopen("size_in.bin", "wb");
	if (fsizein == NULL) {
		printf("Could not sizein.bin!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < num_layers; i++) {
		fwrite(&size_in[i], sizeof(float), 1, fsizein);
	}

	fclose(fsizein);

	// Reading the 1st intermediate output and saving into file ---------------------------------------
	FILE *fsizeout = fopen("size_out.bin", "wb");
	if (fsizeout == NULL) {
		printf("Could not sizeout.bin!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < num_layers; i++) {
		fwrite(&size_out[i], sizeof(float), 1, fsizeout);
	}

	fclose(fsizeout);

	// Reading the 1st intermediate output and saving into file ---------------------------------------
	FILE *fkx = fopen("kernel_x.bin", "wb");
	if (fkx == NULL) {
		printf("Could not kernel_x.bin!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < num_layers; i++) {
		fwrite(&kernel_x[i], sizeof(float), 1, fkx);
	}

	fclose(fkx);

	// Reading the 1st intermediate output and saving into file ---------------------------------------
	FILE *fky = fopen("kernel_y.bin", "wb");
	if (fky == NULL) {
		printf("Could not kernel_y.bin!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < num_layers; i++) {
		fwrite(&kernel_y[i], sizeof(float), 1, fky);
	}

	fclose(fky);

	// Reading the 1st intermediate output and saving into file ---------------------------------------
	FILE *ffinal = fopen("prob_map.txt", "w");
	if (ffinal == NULL) {
		printf("Could not open prob_map.txt!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < IN_R * IN_C; i++) {
		fprintf(ffinal, "%f\n", prob_map[i]);
	}

	fclose(ffinal);
	printf("Finished output!\n");
}

// void wrapper(int16_t input[SAMPLES * ALINES],
// 			 uint8_t env[SAMPLES / DEC_FACTOR * ALINES],
// 			 uint8_t LUT[POWER11],
// 			 char *PARAM_DATA) {

// 	demod(input, env, LUT);
// 	load(PARAM_DATA);
// 	decimation(env);

// 	clock_t start = clock(), diff;
// 	float *prob_map = convolution();
// 	diff = clock() - start;
// 	int msec = diff * 1000 / CLOCKS_PER_SEC;
// 	printf("Convolution took %d seconds %d milliseconds\n", msec/1000, msec%1000);

// 	// make_files();
// 	free_memory();
// }