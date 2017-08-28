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
#include <assert.h>
#include "dnn_conv.h"
#include "demod.h"

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
	float temp1, temp2, temp3, temp4, temp5;

	int pad_zeros = (kernel_x - 1) / 2; // NOTE: Valid only if kernel_x is odd

	conv_kernel_1:for (int y = -2; y <= 2; y++) {
		y_coor = center_y + 2;
		x_coor = center_x - 2;
		if ((x_coor >= 0 && x_coor < IN_C) && (y_coor >= 0 && y_coor < IN_R)) {
			temp1 = param->weights[lyr][img][kernel][((y + pad_zeros) * kernel_x) + (-2 + pad_zeros)] *
					param->output[lyr][img][IN_C * y_coor + x_coor];
		}

		x_coor = center_x - 1;
		if ((x_coor >= 0 && x_coor < IN_C) && (y_coor >= 0 && y_coor < IN_R)) {
			temp2 = param->weights[lyr][img][kernel][((y + pad_zeros) * kernel_x) + (-1 + pad_zeros)] *
					param->output[lyr][img][IN_C * y_coor + x_coor];
		}

		x_coor = center_x;
		if ((x_coor >= 0 && x_coor < IN_C) && (y_coor >= 0 && y_coor < IN_R)) {
			temp3 = param->weights[lyr][img][kernel][((y + pad_zeros) * kernel_x) + (pad_zeros)] *
					param->output[lyr][img][IN_C * y_coor + x_coor];
		}

		x_coor = center_x + 1;
		if ((x_coor >= 0 && x_coor < IN_C) && (y_coor >= 0 && y_coor < IN_R)) {
			temp4 = param->weights[lyr][img][kernel][((y + pad_zeros) * kernel_x) + (1 + pad_zeros)] *
					param->output[lyr][img][IN_C * y_coor + x_coor];
		}

		x_coor = center_x + 2;
		if ((x_coor >= 0 && x_coor < IN_C) && (y_coor >= 0 && y_coor < IN_R)) {
			temp5 = param->weights[lyr][img][kernel][((y + pad_zeros) * kernel_x) + (2 + pad_zeros)] *
					param->output[lyr][img][IN_C * y_coor + x_coor];
		}

		sum += temp1 + temp2 + temp3 + temp4 + temp5;

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

	//int i_idx;

	conv_image_1:for (int r = 0; r < IN_R; r++) {

		conv_image_2:for (int c = 0; c < IN_C; c++) {
			int i_idx = IN_C * r + c;
			param->output[lyr + 1][kernel][i_idx] += conv_kernel(param, kernel_x, kernel_y,
														 c, r, lyr, img, kernel);
		}
	}
}

void single_layer(struct CNN *param,
				  float input[IN_R * IN_C],
				  const int layer) {
	single_layer_1:for (int i = 0; i < IN_R * IN_C; i++) {
		param->output[0][0][i] = input[i];
	}

	// single_layer_2:for (i = 0; i < 5; i++) {
	// 	single_layer_3:for (int j = 0; j < 1; j++) {
	// 		conv_image(param, param->kernel_x[layer], param->kernel_y[layer], layer, j, i);
	// 	}
	// }

	conv_image(param, param->kernel_x[layer], param->kernel_y[layer], layer, 0, 0);
	conv_image(param, param->kernel_x[layer], param->kernel_y[layer], layer, 0, 1);
	conv_image(param, param->kernel_x[layer], param->kernel_y[layer], layer, 0, 2);
	conv_image(param, param->kernel_x[layer], param->kernel_y[layer], layer, 0, 3);
	conv_image(param, param->kernel_x[layer], param->kernel_y[layer], layer, 0, 4);
}
	
