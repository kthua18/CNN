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

// DEBUGGING ONLY
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
static float **buffer, **output; // buffers to hold intermediate outputs
static float *test_kernel;
static float **test_output;
static float *test_image;


// DEBUGGING ONLY
// void debug_print(const char *message, va_list args) { printf(message, args); fflush(stdout); }
#define debug_print printf

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

// Copying and overwriting the values from buffer 1 to buffer 2. 
static void overwrite(float** buffer1, float** buffer2) {
	for (int i = 0; i < max_depth; i++) {
		// for (int j = 0; j < IN_X * IN_Y; j++) {
		// 	buffer2[i][j] = buffer1[i][j];
		// }
		memcpy(buffer2, buffer1, IN_X * IN_Y * sizeof(float));
	}
}

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
  	// int *kernel_x, *kernel_y;

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

  	assert(size_in != NULL);
  	assert(size_out != NULL);
  	assert(kernel_x != NULL);
  	assert(kernel_y != NULL);
  	assert(weights != NULL);
  	assert(biases != NULL);

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
  		for (int j = 0; j < size_in[i]; j++) {
  			weights[i][j] = MALLOC(float*, size_out[i]);
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

  	// Allocating memory for buffers
  	buffer = MALLOC(float*, max_depth);
  	output = MALLOC(float*, max_depth);

  	assert(buffer != NULL);
  	assert(output != NULL);

  	for (int i = 0; i < max_depth; i++) {
  		buffer[i] = CALLOC(float, IN_X * IN_Y);
  		output[i] = CALLOC(float, IN_X * IN_Y);
  		
  		assert(buffer[i] != NULL);
  		assert(output[i] != NULL);
  	}

  	debug_print("Setting buffer\n");
  	for (int z = 0; z < max_depth; z++) {
  		for (int i = 0; i < IN_X * IN_Y; i++) {
			buffer[z][i] = 777.0f;
		}
	}
	debug_print("Set buffer\n");
	
	fclose(file);
}

// Takes a 578x40 B-mode image single pointer and reduce it to 48x40.
// Saves value into a new variable
void decimation(int8_t env[BMODE_X * BMODE_Y]) {
	int x = BMODE_X / IN_X;
	int y = BMODE_Y / IN_Y;
	float sum = 0;

	for (int i = 0; i < IN_X * IN_Y; i++) {
		for (int j = 0; j < y; j++) {
			for (int k = 0; k < x; k++) {
				sum += env[(i * x) + (BMODE_X * j) + k];
			}
		}

		buffer[0][i] = sum / (x * y);
		// printf("Ay it's in da buffer");
	}
}

// Performs convolution for 1 window and returns sum.
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
	
	bool printing = kernel_y == 7 && kernel_x == 7 && lyr == 1 && img == 0 && kernel == 0;

	if (printing)
	debug_print(
		"conv_kernel(kernel_x=%d, kernel_y=%d, center_x=%d, center_y=%d, lyr=%d, img=%d, kernel=%d)\n",
		kernel_x,
		kernel_y,
		center_x,
		center_y,
		lyr,
		img,
		kernel);

	printf("center_x = %d, ", center_x);
	printf("center_y = %d, ", center_y);
	for (int y = -pad_zeros; y <= pad_zeros; y++) {
		y_coor = center_y + y;
		printf("y_coor = %d, ", y_coor);
		for (int x = -pad_zeros; x <= pad_zeros; x++) {
			x_coor = center_x + x;
			printf("x_coor = %d, ", x_coor);
			if (!within_bounds(x_coor, y_coor)) {
				continue;
			}
			
			if (printing)
				debug_print("conv_kernel 1\n");
			
			k_idx = ((y + pad_zeros) * kernel_x) + (x + pad_zeros);
			k_val = weights[lyr][img][kernel][k_idx]; // <-- Real code!
			
			if (printing)
				debug_print("conv_kernel 2\n");
			
			// k_val = test_kernel[k_idx]; // <-- Test code
			i_idx = IN_X * y_coor + x_coor;

			if (printing) {
				debug_print("buffer=%p, accessing buffer[%d][%d]\n", buffer, img, i_idx);	
				debug_print("buffer[%d] = %p\n", img, buffer[img]);
			} 
			assert(buffer != NULL);
			assert(buffer[0] != NULL);
			// help me :(
			// printf("Asserts passed");
			i_val = buffer[img][i_idx]; // during first pass, buffer is the original image!
			if (printing) debug_print("conv_kernel 3\n");
			sum += k_val * i_val;
		}
	}
	return sum; 
}

// Performs convolution for 1 image and saves into variable.
void conv_image(const int kernel_x,
				const int kernel_y,
				const int lyr,
				const int img,
			  	const int kernel) {
	int i_idx;

	// debug_print("conv_image(kernel_x=%d, kernel_y=%d, lyr=%d, img=%d, kernel=%d)\n", kernel_x, kernel_y, lyr, img, kernel);

	for (int y = 0; y < 5; y++) {
		for (int x = 0; x < 5; x++) {

			i_idx = IN_X * y + x;

			// debug_print("  loop(x=%d, y=%d)\n", x, y);
			
			output[img][i_idx] += conv_kernel(kernel_x, kernel_y, x, y, 
											  lyr, img, kernel);
		}
	}
}

// Performs convolution for 1 layer.
void conv_layer(const int layer) {
	debug_print("conv_layer(layer=%d)\n", layer);
	for (int i = 0; i < size_out[layer]; i++) {
		for (int j = 0; j < size_in[layer]; j++) {
			conv_image(kernel_x[layer], kernel_y[layer], layer, j, i);
		}
	}	
}

// Performs ReLU on one layer.
void relu(const int layer) {
	for (int i = 0; i < size_out[layer]; i++) {
		for (int j = 0; j < IN_X * IN_Y; j++) {
		// 	if(output[i][j] < 0) {
		// 		output[i][j] = 0;
		// 	}

		output[i][j] = fmaxf(output[i][j], 0);
		}
	}
}

// Adds the bias of one layer, after convolution. 
void add_bias(const int layer) {
	for (int i = 0; i < size_out[layer]; i++) {
		for (int j = 0; j < IN_X * IN_Y; j++) {
			output[i][j] += biases[layer][i];
		}
	}
}

// Performs sigmond on the final output. 
void sigmoid() {
	for (int i = 0; i < IN_X * IN_Y; i++) {
		output[0][i] = 1.0 / (1.0 + expf(-output[0][i])); // <--- Real code
		// test_output[0][i] = 1.0 / (1.0 + expf(-test_output[0][i])); // <--- Test code
	}
}

// Performs entire convolution, from B-mode to probability map. returns
// a pointer to the probability map.
float* convolution() {
	debug_print("Hey, you entered the function\n");
	for (int i = 0; i < num_layers - 1; i++) {
		printf("i = %d\n", i);
		debug_print("You entered the loop, nice!\n");
		conv_layer(i);
		debug_print("Finished conv_layer\n");
		relu(i);
		debug_print("Finished relu\n");
		add_bias(i);
		debug_print("Finished add_bias\n");
		overwrite(output, buffer);
		debug_print("Finished overwrite\n");
	}

	conv_layer(num_layers - 1);
	sigmoid();
	add_bias(num_layers - 1);
	return output[0];

}

void free_memory() {
	for (int i = 0; i < num_layers; i++) {
		free(biases[i]);
		for (int j = 0; j < size_in[i]; j++) {
			for (int k = 0; k < size_out[i]; k++) {
				free(weights[i][j][k]);
			}
		}
	}

	free(size_in);
	free(size_out);
	free(kernel_x);
	free(kernel_y);


	for (int i = 0; i < max_depth; i++) {
		free(buffer[i]);
		free(output[i]);
	}
}

// Ignore this garbage
void test() {
	// test_kernel = (float*) MALLOC(float, 25);
	// for (int i = 0; i < 25; i++) {
	// 	test_kernel[i] = 0.5;
	// }

 //  	float test_data[48 * 40];
	// for (int i = 0; i < 48 * 40; i++) {
	// 	test_data[i] = 10;
	// }

	// for (int i = 0; i < 48 * 40; i++) {
	// 	buffer[0][i] = 10;
	// }
	

 //  	int count = 0;
 //  	printf("Before:\n");
	// for(int i = 0; i < 200; i++) {
	// 	printf("%f ", buffer[0][i]);
	// 	count++;

	// 	if (count == 48) {
	// 		printf("\n\n");
	// 		count = 0;
	// 	}
	// }

 //  	conv_image(5, 5, 0, 0, 0);

 //  	count = 0;
 //  	printf("After:\n");
	// for(int i = 0; i < 200; i++) {
	// 	printf("%f ", output[0][i]);
	// 	count++;

	// 	if (count == 48) {
	// 		printf("\n\n");
	// 		count = 0;
	// 	}
	// }

 //  	count = 0;
 //  	printf("Kernel:\n");
	// for(int i = 0; i < 25; i++) {
	// 	printf("%f ", test_kernel[i]);
	// 	count++;

	// 	if (count == 5) {
	// 		printf("\n");
	// 		count = 0;
	// 	}
	// }

	// test_output = MALLOC(float*, 5);
	// for (int i = 0; i < 5; i++) {
	// 	test_output[i] = MALLOC(float, IN_X * IN_Y);
	// }

	// for (int i = 0; i < IN_X * IN_Y; i++) {
	// 	test_output[0][i] = -5.0 + i*0.01;
 // 	}

 // 	sigmoid();

 // 	for (int i = 0; i < 200; i++) {
 // 		printf("%f  ", test_output[0][i]);
 // 	}

	convolution();
}