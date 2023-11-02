#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include "stb_image_write.h"
#include "stb_image.h"

__global__ void firstkernel(float, float, int, int, int);
__global__ void secondkernel(unsigned char *, float, float, int, int, int);

int main(int argc, char *argv[]) {
    if (argc != 4)
	{
		printf("Usage:\n");
		printf("--------------------------------------------------------------------\n\n");
		printf("rbf filename_out filename_in \n");
		printf("    rows_per_block \n");
        printf("Where rows_per_block is how many rows a block will process\n")
        printf(".. and threads_per_block is 3x of that\n\n")
		printf("--------------------------------------------------------------------\n");
		return(-1);
	}

    //const int n = 10;  // do this filter 10 times
    const char * filename_out = argv[1];
	const char * filename_in = argv[2];
    int ROWS_PER_BLOCK = argv[3];
	float sigma_spatial = 0.03;
	float sigma_range = 0.1;
    int width, height, channels;
    unsigned char *img_h = stbi_load(filename_in, &width, &height, &channels, 0);
    unsigned char *img_d;
    float *img_tmp_d;
    float *map_factor_a_d; 


    if (img == NULL) {
        printf("Error loading the image :(");
        exit(1);
    }
    printf("Loaded image: w=%d, h=%d, c=%d", width, height, channels);

    int TOTAL_THREADS = height; // * 3
    int THREADS_PER_BLOCK = ROWS_PER_BLOCK; // * 3
    int NUM_BLOCKS;
    if (TOTAL_THREADS % THREADS_PER_BLOCK == 0) 
        NUM_BLOCKS = TOTAL_THREADS / THREADS_PER_BLOCK;
    else
        NUM_BLOCKS = (TOTAL_THREADS / THREADS_PER_BLOCK > 0) ? TOTAL_THREADS / THREADS_PER_BLOCK + 1 : 1;
    
    printf("GPU: %d blocks of %d threads each\n", NUM_BLOCKS, THREADS_PER_BLOCK);

    dim3 grid(NUM_BLOCKS, 1, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);

    cudaMalloc((void**) &img_d, height * width * channels * sizeof(char));
    if (!img_d) {
        printf("cannot allocate img_d of %d by %d\n", width, height);
        stbi_image_free(image);
        exit(1);
    }

    cudaMalloc((void**) &img_tmp_d, height * width * channels * sizeof(float));
    if (!img_tmp_d) {
        printf("cannot allocate img_tmp_d of %d by %d\n", width, height);
        stbi_image_free(image);
        cudaFree(img_d);
        exit(1);
    }

    cudaMalloc((void**) &map_factor_a_d, height * width * channels * sizeof(float));
    if (!img_tmp_d) {
        printf("cannot allocate map_factor_a_d of %d by %d\n", width, height);
        stbi_image_free(image);
        cudaFree(img_d);
        cudaFree(img_tmp_d);
        exit(1);
    }

    // for timing
    double elapse = 0;
    clock_t start, end;

    // fire up the timer
    start = clock();

    // copy img to device
    cudaMemCpy(img_d, img_h, height * width * channels * sizeof(char), cudaMemcpyHostToDevice);
    // invoke first kernel
    firstkernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(sigma_spatial, sigma_range, width, height, channels);
    // now we have in device mem: img, img_temp, map_factor_a

    // copy img back
    cudaMemCpy(img_h, img_d, height * width * channels * sizeof(char), cudaMemcpyDeviceToHost);

    // stop timer
    end = clock();
    // calculate time
    elapse = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("GPU version: %lf seconds\n", elapse);
    printf("-------------------\n");

    // write out processed image
    stb_write_jpg(filename_out, width, height, channels, img_h, 100);
    // clear up
    free(img_h);
    cudaFree(img_d);

}

__global__ void firstkernel(float sigma_spatial, float sigma_range, int width, int height, int channels) 
{
    const int wh = width * height;
    const int wc = width * channels;
    const int whc = width * height * channels;

    int row_number = (blockIdx.x * blockDim.x) + threadIdx.x; // which row is this
    //int row_offset = threadIdx.y;

    float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * width)));
    float inv_alpha_ = 1 - alpha;
    float ypr, ypg, ypb, ycr, ycg, ycb;
    float fp, fc;

    float * temp_x = &img_tmp_d[row_number * wc];
    unsigned char *in_x = &img_d[row_number * wc];
    unsigned char *texture_x = &img_d[row_number * wc];
    *temp_x++ = ypr = *in_x++; 
    *temp_x++ = ypg = *in_x++; 
    *temp_x++ = ypb = *in_x++;
    unsigned char tpr = *texture_x++; 
    unsigned char tpg = *texture_x++;
    unsigned char tpb = *texture_x++;

    float *temp_factor_x = &map_factor_d[row_number * width];
    *temp_factor_x++ = fp = 1;

    // from left to right
    for (int x = 1; x < width; x++) 
    {
        unsigned char tcr = *texture_x++; 
        unsigned char tcg = *texture_x++; 
        unsigned char tcb = *texture_x++;
        unsigned char dr = abs(tcr - tpr);
        unsigned char dg = abs(tcg - tpg);
        unsigned char db = abs(tcb - tpb);
        int range_dist = (((dr << 1) + dg + db) >> 2);
        float weight = range_table[range_dist];
        float alpha_ = weight*alpha;
        *temp_x++ = ycr = inv_alpha_*(*in_x++) + alpha_*ypr; 
        *temp_x++ = ycg = inv_alpha_*(*in_x++) + alpha_*ypg; 
        *temp_x++ = ycb = inv_alpha_*(*in_x++) + alpha_*ypb;
        tpr = tcr; tpg = tcg; tpb = tcb;
        ypr = ycr; ypg = ycg; ypb = ycb;
        *temp_factor_x++ = fc = inv_alpha_ + alpha_*fp;
        fp = fc;
    }

    *--temp_x; *temp_x = 0.5f*((*temp_x) + (*--in_x));
    *--temp_x; *temp_x = 0.5f*((*temp_x) + (*--in_x));
    *--temp_x; *temp_x = 0.5f*((*temp_x) + (*--in_x));
    tpr = *--texture_x; 
    tpg = *--texture_x; 
    tpb = *--texture_x;
    ypr = *in_x; ypg = *in_x; ypb = *in_x;

    *--temp_factor_x; *temp_factor_x = 0.5f*((*temp_factor_x) + 1);
    fp = 1;

    // from right to left
    for (int x = width - 2; x >= 0; x--) {
        unsigned char tcr = *--texture_x; 
        unsigned char tcg = *--texture_x; 
        unsigned char tcb = *--texture_x;
        unsigned char dr = abs(tcr - tpr);
        unsigned char dg = abs(tcg - tpg);
        unsigned char db = abs(tcb - tpb);
        int range_dist = (((dr << 1) + dg + db) >> 2);
        float weight = range_table[range_dist];
        float alpha_ = weight * alpha;

        ycr = inv_alpha_ * (*--in_x) + alpha_ * ypr; 
        ycg = inv_alpha_ * (*--in_x) + alpha_ * ypg; 
        ycb = inv_alpha_ * (*--in_x) + alpha_ * ypb;
        *--temp_x; *temp_x = 0.5f*((*temp_x) + ycr);
        *--temp_x; *temp_x = 0.5f*((*temp_x) + ycg);
        *--temp_x; *temp_x = 0.5f*((*temp_x) + ycb);
        tpr = tcr; tpg = tcg; tpb = tcb;
        ypr = ycr; ypg = ycg; ypb = ycb;

        fc = inv_alpha_ + alpha_*fp;
        *--temp_factor_x; 
        *temp_factor_x = 0.5f*((*temp_factor_x) + fc);
        fp = fc;
    }
}    