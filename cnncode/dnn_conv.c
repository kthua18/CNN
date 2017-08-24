/* 
 * dnn_conv.c
 * Implements convolutional neural network (CNN) to be used in the Zynq 7000x 
 * chip. Input is the image of interest, output is a probability map. Convolution
 * is performed in the spatial domain.
 *
 * @author      Kim Hua
 * @version     1.0, 2017/7/31
 *
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "dnn_conv.h"

#define MALLOC(TYPE, COUNT) ((TYPE *) malloc(sizeof(TYPE) * COUNT))
#define CALLOC(TYPE, COUNT) ((TYPE *) calloc(COUNT, sizeof(TYPE)))

static int num_layers; // number of layers in CNN
static int max_depth = 0; // maximum depth (input images) of all the layers
static int *size_in, *size_out; // the input and output depths of the kernels
static int *kernel_x, *kernel_y; // kernel dimensions for all the layers
static float ****weights, **biases; // matrix of kernel and bias values
// static float **buffer, **output; // buffers to hold intermediate outputs
float ***output;
float *prob_map;

static float *test_kernel;
static float **test_output;
static float *test_image;




/*
 * \brief Re-order the kernel data
 *
 * \param[in] buffer pointer to the original kernel data
 * \param[out] kbuffer pointer to the re-ordered kernel data
 * \param[in] kernel_x horizontal kernel size
 * \param[in] kernel_y vertical kernel size
 * \param[in] num_kernels number of kernels
 *
 * This function re-orders kernel data (Tensorflow format) to the format
 * used in the DNN C module. 
 */

static void reorder_kernel(const float *buffer,
                           const int layer_idx,
                           const int in,
                           const int out,
                           const int total_kernels,
                           const int kernel_size) {
	int i, j, k;
	for (i = 0; i < in; i++) {
		for (j = 0; j < out; j++) {
			for (k = 0; k < kernel_size; k++) {
				weights[layer_idx][i][j][k] = buffer[k * total_kernels + i * in + j];
			}
		}
	}
}

// Returns true if a coordinate is within the bounds of the image, 
// false otherwise.
bool within_bounds(const int x, const int y) {
	return (x >= 0 && x < IN_X) &&
		   (y >= 0 && y < IN_Y);
}

// // Copying and overwriting the values from buffer 1 to buffer 2. 
// static void overwrite(float** buffer1, float** buffer2) {
// 	for (int i = 0; i < max_depth; i++) {
// 		memcpy(buffer2, buffer1, IN_X * IN_Y * sizeof(float));
// 	}
// }

// Reads the file; allocates appropiate memory for kernel_x, kernel_y,
// size_in, size_out, weights, and biases; populates the arrays in the
// appropiate format.
//  
// kernel_x, kernel_y: the x and y dimensions to describe shape of kernel.
// size_in, size_out: the input and output depth of each layer.
// weights: 4D array for layers, images, kernels, and values.
// biases: 2D array for layers, images.
// buffer1, buffer2: 2D array of buffers used to perform convolution. 
void load(const char *filename) {
	FILE *file = fopen(filename, "rb");
  	int idx_layer;

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
	prob_map = CALLOC(float, IN_X * IN_Y);

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
  			output[i][j] = CALLOC(float, IN_X * IN_Y); 
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


  	// // Allocating memory for buffers
  	// buffer = MALLOC(float*, max_depth);
  	// output = MALLOC(float*, max_depth);

  	// for (int i = 0; i < max_depth; i++) {
  	// 	buffer[i] = CALLOC(float, IN_X * IN_Y);
  	// 	output[i] = CALLOC(float, IN_X * IN_Y);
  	// }

	fclose(file);
}

// Takes a 578x40 B-mode image single pointer and reduce it to 48x40.
// Saves values into buffer.
void decimation(uint8_t env[BMODE_X * BMODE_Y]) {
	int x = BMODE_X / IN_X; // 576 / 48 = 12
	int y = BMODE_Y / IN_Y; // 80 / 40 = 2
	int sum = 0;
	static float *buffer;
	buffer = MALLOC(float, IN_X * IN_Y);

	// for (int i = 0; i < IN_X * IN_Y; i++) {
	// 	for (int j = 0; j < 2; j++) {
	// 		for (int k = 0; k < 12; k++) {
	// 			sum += env[(i * 12) + (BMODE_X * j) + k];
	// 		}
	// 	}
	// 	// buffer[0][i] = sum / (x * y);
	// 	buffer[i] = (float) sum / (x * y);
	// 	//printf("i_val = %f, ", output[0][0][i]);
	// 	sum = 0;
	// }
	int count = 0;

	for (int i = 0; i < IN_Y; i++) {
		for (int j = 0; j < IN_X; j++) {
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

	for (int i = 0; i < IN_X; i++) {
		for (int j = 0; j < IN_Y; j++) {
			output[0][0][IN_Y * i + j] = buffer[IN_X * j + i];
		}
	}

	for (int i = 0; i < IN_X * IN_Y; i++) {
		output[0][0][i] = output[0][0][i] / 255.0;
	}
}

// Performs convolution for 1 window and returns sum.
float conv_kernel(const int kernel_x,
				  const int kernel_y,
				  const int center_x,
				  const int center_y,
				  const int lyr,
				  const int img, // size_in
			  	  const int kernel) { // size_out
	// printf("You have entered conv_kernel!\n");
	int x_coor; 
	int y_coor;
	int k_idx, i_idx;
	float k_val, i_val;
	float sum = 0;
	int pad_zeros = (kernel_x - 1) / 2; // NOTE: Valid only if kernel_x is odd

	int countx = 0;
	int county = 0;

	for (int y = -pad_zeros; y <= pad_zeros; y++) {
		y_coor = center_y + y;
		for (int x = -pad_zeros; x <= pad_zeros; x++) {
			x_coor = center_x + x;
			if (!within_bounds(x_coor, y_coor)) {
				continue;
			}
			

			k_idx = ((y + pad_zeros) * kernel_x) + (x + pad_zeros);
			k_val = weights[lyr][img][kernel][k_idx]; // <-- Real code!
			// k_val = test_kernel[k_idx]; // <-- Test code
			i_idx = IN_X * y_coor + x_coor;
			// i_val = buffer[img][i_idx]; // during first pass, buffer is the original image!
			
			i_val = output[lyr][img][i_idx]; 



			// if (countx < 200) {
			// 	printf("k_idx = %d, i_idx = %d\n", k_idx, i_idx);
			// 	countx++;
			// }

			// printf("k = %d, i = %d  ", k_idx, i_idx);
			
			sum += k_val * i_val;
		}
	}
	return sum; 
}

// Performs convolution for 1 image and saves into variable.
void conv_image(const int kernel_x,
				const int kernel_y,
				const int lyr,
				const int img, // suze in
			  	const int kernel) { // size out
	//printf("Entering conv_image!\n");
	int i_idx;
	if (lyr == 7) {
		//printf("kernel_x = %d\nkernel_y = %d\nlyr = %d\nimg = %d\nkernel = %d\n", kernel_x,
		//	kernel_y, lyr, img, kernel);
	}
	for (int y = 0; y < IN_Y; y++) {
		for (int x = 0; x < IN_X; x++) {
			i_idx = IN_X * y + x;
			// output[img][i_idx] += conv_kernel(kernel_x, kernel_y, x, y, 
			// 								  lyr, img, kernel);
			if (lyr < num_layers - 1) {
				// assert(output[lyr + 1][img] != NULL);
				output[lyr + 1][img][i_idx] += conv_kernel(kernel_x, kernel_y, x, y, 
											  lyr, img, kernel);
			

			} else {
				// printf("Doing last conv_kernel!");
				prob_map[i_idx] += conv_kernel(kernel_x, kernel_y, x, y, 
											  lyr, img, kernel);
			}
			// printf("x = %d, ", x);

		}
		// printf("y = %d, ", y);
	}
	// printf("Exiting conv_image!\n");
}

// Performs convolution for 1 layer.
void conv_layer(const int layer) {
	// printf("You have entered conv_layer!\n");
	for (int i = 0; i < size_out[layer]; i++) {
		for (int j = 0; j < size_in[layer]; j++) {
			// printf("You have gone in the conv_layer loop! i = %d, j = %d\n", i, j);
			conv_image(kernel_x[layer], kernel_y[layer], layer, j, i);
		}
	}	
}

// Performs ReLU on one layer.
void relu(const int layer) {
	for (int i = 0; i < size_out[layer]; i++) {
		for (int j = 0; j < IN_X * IN_Y; j++) {
			// output[i][j] = fmaxf(output[i][j], 0);
			output[layer + 1][i][j] = fmaxf(output[layer + 1][i][j], 0);
			
		}
	}
}

// Adds the bias of one layer, after convolution. 
void add_bias(const int layer) {
	if (layer == 7) {
		for (int i = 0; i < IN_X * IN_Y; i++) {
			// printf("prob_map i =%f", prob_map[i]);
			prob_map[i] += biases[layer][0];
		}
	} else {
		for (int i = 0; i < size_out[layer]; i++) {
			for (int j = 0; j < IN_X * IN_Y; j++) {
				output[layer + 1][i][j] += biases[layer][i]; 
			}
		}
	}
}

// Performs sigmond on the final output. 
void sigmoid() {
	for (int i = 0; i < IN_X * IN_Y; i++) {
		// output[0][i] = 1.0 / (1.0 + expf(-output[0][i])); // <--- Real code
		// test_output[0][i] = 1.0 / (1.0 + expf(-test_output[0][i])); // <--- Test code
		prob_map[i] = 1.0 / (1.0 + expf(-prob_map[i]));
	}
}

// Performs entire convolution, from B-mode to probability map. returns
// a pointer to the probability map.
float* convolution() {
	printf("You got to convolution!\n");
	for (int i = 0; i < num_layers - 1; i++) {
		printf("You entered the loop, i = %d\n", i);
		conv_layer(i);
		printf("You have done conv_layer!\n");
		add_bias(i);
		printf("You have done add_bias!\n");
		relu(i);
		printf("You have done relu!\n");
		
		// overwrite(output, buffer);
	}
	printf("Doing last layer!\n");
	conv_layer(num_layers - 1);
	printf("You have done conv_layer!\n");
	add_bias(num_layers - 1);
	printf("You have done add_bias!\n");
	sigmoid();
	printf("You have done sigmoid!\n");
	
	// free_memory();
	return prob_map;
}

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
}

// Ignore this garbage
void test() {

	for (int i = 0; i < IN_X * IN_Y; i++) {
		printf("%f ", output[0][0][i]);
	}
}

void make_files() {
	FILE *fweights = fopen("weights.txt", "w");
	if (fweights == NULL) {
		printf("Could not open pfweights.txt!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < num_layers; i++) {
		for (int j = 0; j < size_in[i]; j++) {
			for (int k = 0; k < size_out[i]; k++) {
				for (int l = 0; l < kernel_x[i]*kernel_y[i]; l++) {
					fprintf(fweights, "%f\n", weights[i][j][k][l]);
				}
			}
		}
	}

	fclose(fweights);
	printf("Finished weights!\n");

	FILE *fbiases = fopen("biases.txt", "w");
	if (fbiases == NULL) {
		printf("Could not open biases.txt!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < num_layers; i++) {
		for (int j = 0; j < size_out[i]; j++) {
			fprintf(fbiases, "%f\n", biases[i][j]);
		}
	}

	fclose(fbiases);
	printf("Finished biases!\n");

	FILE *fenv = fopen("dec_env.txt", "w");
	if (fenv == NULL) {
		printf("Could not dec_env.txt!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < IN_X * IN_Y; i++) {
		fprintf(fenv, "%f\n", output[0][0][i]);
	}

	fclose(fenv);
	printf("Finished dec_env!\n");

	FILE *fout = fopen("output.txt", "w");
	if (fout == NULL) {
		printf("Could not output.txt!");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < IN_X * IN_Y; j++) {
			fprintf(fout, "%f\n", output[1][i][j]);
		}
	}

	fclose(fout);
	printf("Finished output!\n");
	printf("Jizz in my pants!\n");
}