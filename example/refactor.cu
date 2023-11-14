#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include "stb_image_write.h"
#include "stb_image.h"

__global__ void alphasKernel(float, float, int, int, int);
__global__ void roundOneKernel(unsigned char *, float, float, int, int, int);
// __global__ void roundTwoKerkel()

int main(int argc, char *argv[]) {
    if (argc != 4)
	{
		printf("Usage:\n");
		printf("--------------------------------------------------------------------\n\n");
		printf("rbf filename_out filename_in \n");
		printf("    rows_per_block \n");
        printf("Where rows_per_block is how many rows a block will process\n\n")
		printf("--------------------------------------------------------------------\n");
		return(-1);
	}

    
    const char * filename_out = argv[1];
	const char * filename_in = argv[2];
    int ROWS_PER_BLOCK = argv[3];
    int threads_per_block, num_blocks, total_threads;
	float sigma_spatial = 0.03;
	float sigma_range = 0.1;
    int width, height, channels;

    unsigned char *img_h = stbi_load(filename_in, &width, &height, &channels, 0);
    unsigned char *img_d;
    float *img_tmp_d;
    float *map_factor_a_d;
    float *map_factor_b_d;
    float *map_factor_c_d;
    float *slice_factor_a_d;
    float *slice_factor_b_d;
    float *line_factor_a_d;
    float *line_factor_b_d;


    if (img == NULL) {
        printf("Error loading the image :(");
        exit(1);
    }
    printf("Loaded image: w=%d, h=%d, c=%d", width, height, channels);


    /* -------------------------------------------------------------- */
    /* Prepare for alphasKernel                                       */
    /* Reads from: img_d; Writes to: map_factor_b_d, map_factor_c_d   */
    /* -------------------------------------------------------------- */

    threads_per_block = ROWS_PER_BLOCK * (width - 1);
    total_threads = width * height;
    if (total_threads% threads_per_block == 0) 
        num_blocks = total_threads / threads_per_block;
    else
        num_blocks = (total_threads / threads_per_block > 0) ? total_threads / threads_per_block + 1 : 1;
    printf("GPU prelim kernel: %d blocks of %d threads each\n", num_blocks, threads_per_block);

    cudaMalloc((void**) &img_d, height * width * channels * sizeof(char));
    if (!img_d) {
        printf("Cannot allocate buffer img_d on device\n")
        stbi_image_free(image);
        exit(1);
    }

    cudaMalloc((void**) &map_factor_b_d, height * width * sizeof(float));
    if (!img_d) {
        printf("Cannot allocate buffer map_factor_b_d on device\n")
        stbi_image_free(image);
        cudaFree(img_d);
        exit(1);
    }

    cudaMalloc((void**) &map_factor_c_d, height * width * sizeof(float));
    if (!img_d) {
        printf("Cannot allocate buffer map_factor_c_d on device\n")
        stbi_image_free(image);
        cudaFree(img_d);
        cudaFree(map_factor_b_d);
        exit(1);
    }
    // all allocations done
    // threadIdx.x = which row, threadIdx.y = column former


}

__global__ void alphasKernel(
    float sigma_spatial, float sigma_range, 
    int width, int height, int channel) {
        __shared__ float range_table[256] = {0};    // look up table will be computed as needed

        float inv_sigma_range = 1.0f / (sigma_range * 255);
        float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * width)));


}