#include <stdio.h> // for reading files
#include <stdlib.h> // for malloc etc.
#include <stdint.h> // for fixed-width int types
#include "demod.h"

void demod(int16_t input[SAMPLES * ALINES], uint8_t env[SAMPLES / DEC_FACTOR * ALINES], uint8_t LUT[POWER11]) {

	
	int16_t output[SAMPLES * ALINES];
	int sum = 0;
	int count = 0;
	int index = 0;

/*
	// Interpret raw binary data; two's complement
	for(i = 0; i < SAMPLES; i++) {
		if (input[i] > POWER11) {
			output[i] = abs(input[i] - POWER12);
		} else {
			output[i] = input[i];
		}

		sum = sum + output[i];
		count++;
		if (count == DEC_FACTOR) {
			
			temp = sum / 10;
			env[index] = LUT[temp + 1];

			index++;
			count = 0;
			sum = 0;
		}
	}
*/

	// Interpret raw data, two's complement
	for(int i = 0; i < SAMPLES * ALINES; i++) {
		if (input[i] > POWER11) {
			output[i] = abs(input[i] - POWER12);
		} else {
			output[i] = input[i];
		}
	}

	// Segment 
	for(int i = 0; i < SAMPLES * ALINES; i++) {
		sum = sum + output[i];
		count++; 
		if (count == DEC_FACTOR) {
			sum = sum / DEC_FACTOR;
			// printf("%u ", sum);
			env[index] = LUT[sum + 1];

			index++;
			count = 0;
			sum = 0;
		}
	}

}



// uint8_t **demod(const char *filename) {
// 	FILE *f = fopen(filename, "rb");
// 	printf("Hi");

// 	// Allocate one pointer for each row of the image
// 	uint8_t **data = (uint8_t**) malloc(sizeof(uint8_t*) * HEIGHT);
// 	if (data == NULL) {
// 		perror("failed to allocate memory, the world is on fire");
// 		exit(EXIT_FAILURE);
// 	}

// 	// Fill each pointer with an allocated array of bytes.
// 	for (int i = 0; i < HEIGHT; i++) {
// 		data[i] = (uint8_t*) malloc(sizeof(uint8_t) * WIDTH);
// 	}

// 	uint16_t buf;
// 	// Read a single 2-byte quantity from file, storing the resulting data in buf.
// 	szie_t res = fread(&buf, 2, 1, f);
// 	while (!feof(f)) {
// 		//printf("%d\n", buf);
// 		res = fread(&buf, 2, 1, f);
// 	}
// }
