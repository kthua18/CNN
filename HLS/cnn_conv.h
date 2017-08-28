#ifndef _DNN_CONV_H_
#define _DNN_CONV_H_

#define IN_R 48 // Size of image used for convolution, rows
#define IN_C 40 // Columns
#define BMODE_X 576 // Size of BMode image
#define BMODE_Y 80
#define NUM_LAYERS 8
#define MAX_SIZE 5
#define MAX_KERNEL 17

#define SAMPLES 5760
#define ALINES 80
#define DEC_FACTOR 10
#define BYTES_PER_SAMPLE 2
#define POWER11 2048
#define POWER12 4096

struct CNN {
   int size_in[NUM_LAYERS];
   int size_out[NUM_LAYERS];
   int kernel_x[NUM_LAYERS];
   int kernel_y[NUM_LAYERS];
   float weights[NUM_LAYERS][MAX_SIZE][MAX_SIZE][MAX_KERNEL * MAX_KERNEL];
   float biases[NUM_LAYERS][MAX_SIZE];
   float output[NUM_LAYERS][MAX_SIZE][IN_R * IN_C];
   float prob_map[IN_R * IN_C];
};

bool within_bounds(const int x, const int y);

float conv_kernel(struct CNN *param,
				  const int kernel_x,
				  const int kernel_y,
				  const int center_x,
				  const int center_y,
				  const int layer,
				  const int kernel,
				  const int index);

void conv_image(struct CNN *param,
				const int kernel_x,
				const int kernel_y,
				const int lyr,
				const int img,
			  	const int kernel);

void conv_layer(struct CNN *param, const int layer);

void relu(struct CNN *param,
		  const int layer);

void add_bias(struct CNN *param,
			  const int layer);		  

void sigmoid(struct CNN *param);

void convolution(struct CNN *param, float input[IN_R * IN_C]);

void single_layer(struct CNN *param,
				  float input[IN_R * IN_C],
				  float test_output[2][5][IN_R * IN_C],
				  const int layer);

// void test(struct CNN *param,
// 		  const int layer);

#endif
