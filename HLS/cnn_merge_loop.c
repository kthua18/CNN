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
#include <assert.h>
#include "dnn_conv.h"
#include "demod.h"
#include "global_var.h"

#define MALLOC(TYPE, COUNT) ((TYPE *) malloc(sizeof(TYPE) * COUNT))
#define CALLOC(TYPE, COUNT) ((TYPE *) calloc(COUNT, sizeof(TYPE)))
#define PARAM_DATA "neural_net_parameters.bin"

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
 * Performs convolution for 1 window and returns the value.
 *
 * kernel_x, kernel_y: The number of columns and rows of the kernel.
 * center_x, center_y: The center "coordinates" of the image.
 * lyr: The current layer to be calculated.
 * img: The current image to be calculated, or size_in.
 * kernel: The current kernel to be calculated, or size_out.
 */
float conv_kernel(struct CNN *param,
				  const int kernel_x,
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
	assert(pad_zeros == 2);

	conv_kernel_1:for (int y = -2; y <= 2; y++) {

		y_coor = center_y + y; 
		conv_kernel_2:for (int x = -2; x <= 2; x++) {

			x_coor = center_x + x;
			if (!within_bounds(x_coor, y_coor)) {
				continue;
			}
			k_idx = ((y + pad_zeros) * kernel_x) + (x + pad_zeros);
			k_val = param->weights[lyr][img][kernel][k_idx]; 
			i_idx = IN_C * y_coor + x_coor;
			i_val = param->output[lyr][img][i_idx]; 
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
void conv_image(struct CNN *param,
				const int kernel_x,
				const int kernel_y,
				const int lyr,
				const int img, 
			  	const int kernel) { 

	int i_idx;

	conv_image_1:for (int r = 0; r < IN_R; r++) {
		conv_image_2:for (int c = 0; c < IN_C; c++) {
			i_idx = IN_C * r + c;
			if (lyr < NUM_LAYERS - 1) {

				param->output[lyr + 1][kernel][i_idx] += conv_kernel(param, kernel_x, kernel_y, 
														 c, r, lyr, img, kernel);
			// If on final layer, save values into prob_map.
			} else {
				param->prob_map[i_idx] += conv_kernel(param, kernel_x, kernel_y, c, r, 
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
void conv_layer(struct CNN *param,
				const int layer) {
	// param->size_out[layer]
	conv_layer_1:for (int i = 0; i < param->size_out[layer]; i++) {
		conv_layer_2:for (int j = 0; j < param->size_in[layer]; j++) {
			conv_image(param, param->kernel_x[layer], param->kernel_y[layer], layer, j, i);
		}
	}	
}

void single_layer(struct CNN *param,
				  float input[IN_R * IN_C],
				  const int layer) {
	
	single_layer_1:for (int i = 0; i < IN_R * IN_C; i++) {
		param->output[0][0][i] = input[i];
	}

	single_layer_2:for (i = 0; i < 5; i++) {
		single_layer_3:for (int j = 0; j < 1; j++) {
			conv_layer_1:for (int i = 0; i < param->size_out[layer]; i++) {
				conv_layer_2:for (int j = 0; j < param->size_in[layer]; j++) {
					int i_idx;

					conv_image_1:for (int r = 0; r < IN_R; r++) {
						conv_image_2:for (int c = 0; c < IN_C; c++) {
							int x_coor; 
							int y_coor;
							int k_idx, i_idx;
							float k_val, i_val;
							float sum = 0;

							int pad_zeros = (kernel_x - 1) / 2;

							conv_kernel_1:for (int y = -2; y <= 2; y++) {
								y_coor = center_y + y; 
								conv_kernel_2:for (int x = -2; x <= 2; x++) {
									x_coor = center_x + x;
									if (!within_bounds(x_coor, y_coor)) {
										continue;
									}
									k_idx = ((y + pad_zeros) * kernel_x) + (x + pad_zeros);
									k_val = param->weights[lyr][img][kernel][k_idx]; 
									i_idx = IN_C * y_coor + x_coor;
									i_val = param->output[lyr][img][i_idx]; 
									sum += k_val * i_val;
								}
							}
							param->output[lyr + 1][kernel][i_idx] += sum;
						}
					}
				}
			}	
		}
	}
}


/*
 * Performs ReLU on one layer, and overwrites the value at that
 * location in the variable output. 
 *
 * layer: The current layer to be calculated.
 */
void relu(struct CNN *param,
		  const int layer) {
	relu_1:for (int i = 0; i < param->size_out[layer]; i++) {
		relu_2:for (int j = 0; j < IN_R * IN_C; j++) {
			param->output[layer + 1][i][j] = fmaxf(param->output[layer + 1][i][j], 0);
			
		}
	}
}

/*
 * Adds the bias of one layer, and overwrites the result at that location
 * in output. If the final layer is reached, adds the bias to prob_map.
 *
 * layer: The current layer to be calculated 
 */
void add_bias(struct CNN *param,
			  const int layer) {
	if (layer < NUM_LAYERS - 1) {
		add_bias_1:for (int i = 0; i < param->size_out[layer]; i++) {
			add_bias_2:for (int j = 0; j < IN_R * IN_C; j++) {
				param->output[layer + 1][i][j] += param->biases[layer][i]; 
			}
		}
	// If on final layer, save values into prob_map.
	} else {
		add_bias_3:for (int i = 0; i < IN_R * IN_C; i++) {
			param->prob_map[i] += param->biases[layer][0];
		}
	}
}

/*
 * Performs sigmond on the prob_map.
 */
void sigmoid(struct CNN *param) {
	sigmoid_1:for (int i = 0; i < IN_R * IN_C; i++) {
		param->prob_map[i] = 1.0 / (1.0 + expf(-param->prob_map[i]));
	}
}

/*
 * Performs entire convolution, from B-mode to probability map. Returns
 * a pointer to the probability map.
 */
void convolution(struct CNN *param, float input[IN_R * IN_C]) {
	// Start test!
	
	convolution_1:for (int i = 0; i < IN_R * IN_C; i++) {
		param->output[0][0][i] = input[i];
	}
	// End test!
	convolution_2:for (i = 0; i < NUM_LAYERS - 1; i++) {
		
		// Timing each layer of the convolution
		conv_layer(param, i);
		add_bias(param, i);
		relu(param, i);
	}
	conv_layer(param, NUM_LAYERS - 1);
	
	add_bias(param, NUM_LAYERS - 1);

	sigmoid(param);
}


// // Ignore everything below this
// void test(struct CNN *param,
// 		  const int layer) {
// 	for (int i = 0; i < IN_R * IN_C; i++) {
// 		printf("%f ", param->output[0][0][i]);
// 	}

// 	FILE *fout = fopen("output0.bin", "wb");
// 	if (fout == NULL) {
// 		printf("Could not open weights.bin!");
// 		exit(EXIT_FAILURE);
// 	}

// 	for (int i = 0; i < param->size_out[layer]; i++) {
// 		for (int j = 0; j < IN_C * IN_R; j++) {
// 			// fprintf(fweights, "%f\n", weights[i][j][k][l]);
// 			fwrite(&param->output[layer][i][j], sizeof(float), 1, fout);
			
// 		}
// 	}
// 	fclose(fout);
// 	printf("Finished output0!\n");





	
