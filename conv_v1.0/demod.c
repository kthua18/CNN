#include <stdio.h> // for reading files
#include <stdlib.h> // for malloc etc.
#include <stdint.h> // for fixed-width int types
#include "demod.h"

void demod(int16_t input[SAMPLES * ALINES], uint8_t env[SAMPLES / DEC_FACTOR * ALINES], uint8_t LUT[POWER11]) {

	
	int16_t output[SAMPLES * ALINES];
	int sum = 0;
	int count = 0;
	int index = 0;

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