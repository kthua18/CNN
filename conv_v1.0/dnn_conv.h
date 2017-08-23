#ifndef _DNN_CONV_H_
#define _DNN_NETWORK_H_

#define IN_R 48 // Size of image used for convolution, rows
#define IN_C 40 // Columns
#define BMODE_X 576 // Size of BMode image
#define BMODE_Y 80

void load(const char *filename);
bool within_bounds(const int x, const int y);
float conv_kernel(const int kernel_x,
				  const int kernel_y,
				  const int center_x,
				  const int center_y,
				  const int layer,
				  const int kernel,
				  const int index);

void conv_image(const int kernel_x,
				const int kernel_y,
				const int lyr,
				const int img,
			  	const int kernel);

void test();

void conv_layer(const int layer);

void decimation(uint8_t env[BMODE_X * BMODE_Y]);

void add_bias(const int layer);

void relu(const int layer);

float* convolution();

void free_memory();

void make_files();

#endif