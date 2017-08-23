// demod.h file
#ifndef _demod_h
#define _demod_h

// static const int16_t SAMPLES = 5760;
// static const int16_t ALINES = 80;
// static const int16_t DEC_FACTOR = 10;
// static const int16_t BYTES_PER_SAMPLE = 2;
// static const int16_t POWER11 = 2048;
// static const int16_t POWER12 = 4096;

void demod(int16_t input[SAMPLES * ALINES], uint8_t env[SAMPLES / DEC_FACTOR * ALINES], uint8_t LUT[POWER11]);

#endif