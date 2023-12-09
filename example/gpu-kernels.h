#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include <stdio.h>
#include <stdlib.h>

void invokeNaiveKernel(unsigned char* img, int width, 
    int height, int channel, float sigma_patial, 
    float sigma_range, int rows_per_block, float* buffer);


void refactorGPU(unsigned char* img_h, int width, int height, int channel,
    float sigma_spatial, float sigma_range, int rows_per_block, float* buffer);   

#endif