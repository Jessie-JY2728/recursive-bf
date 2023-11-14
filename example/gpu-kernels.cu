#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#define QX_DEF_CHAR_MAX 255

___global___ void naiveKernel(unsigned char*, float*, float*, float* , int, int, int, float);
___global___ void prelimKernel();
___global___ void firstKernel();
___global___ void secondKernel();

void naiveRemainder();

void invokeNaiveKernel(
    unsigned char* img_h, int width, int height, int channel,
    float sigma_spatial, float sigma_range, int rows_per_block
    ) 
{
    unsigned char* img_d;
    float *img_tmp_d;   // img_temp, on device
    float *map_factor_a_d;  // map_factor_a, on device
    float *map_factor_a_h;  // map_factor_a, on host, copied back from device
    float *img_tmp_h;   // img_temp, copied back from device
    float *range_table_d;

    // range table for look up
    float range_table[QX_DEF_CHAR_MAX + 1];
    float inv_sigma_range = 1.0f / (sigma_range * QX_DEF_CHAR_MAX);
    for (int i = 0; i <= QX_DEF_CHAR_MAX; i++) 
        range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));

    // initialize on host side
    img_tmp_h = new float[width * height * channel];
    map_factor_a_h = new float[width * height];

    // initialize on device using cudaMalloc
    cudaMalloc((void**) &range_table_d, (QX_DEF_CHAR_MAX + 1) * sizeof(float));
    if (!range_table_d) {
        printf("Naive Kernel: Cuda malloc fail on range_table_d");
        delete[] img_tmp_h;
        delete[] map_factor_a_h;
        exit(1);
    }
    cudaMalloc((void**) &img_d, height * width * channel * sizeof(char));
    if (!img_d) {
        printf("Naive Kernel: Cuda malloc fail on img_d");
        cudaFree(range_table_d);
        delete[] img_tmp_h;
        delete[] map_factor_a_h;
        exit(1);
    }

    cudaMalloc((void**) &img_tmp_d, height * width * channel * sizeof(float));
    if (!img_tmp_d) {
        printf("Naive Kernel: Cuda malloc fail on img_tmp_d");
        delete[] img_tmp_h;
        delete[] map_factor_a_h;
        cudaFree(img_d);
        cudaFree(range_table_d);
        exit(1);
    }

    cudaMalloc((void**) &map_factor_a_d, height * width * channel * sizeof(float));
    if (!map_factor_a_d) {
        printf("Naive Kernel: Cuda malloc fail on map_factor_a_d");
        delete[] img_tmp_h;
        delete[] map_factor_a_h;
        cudaFree(img_d);
        cudaFree(img_tmp_d);
        cudaFree(range_table_d);
        exit(1);
    } // finish device side initialization

    // kernel params
    int total_threads = height;
    int threads_per_block = rows_per_block;
    int num_blocks;
    if (total_threads % threads_per_block == 0) 
        num_blocks = total_threads / threads_per_block;
    else
        num_blocks = total_threads / threads_per_block + 1;

    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    // copy input image to device
    cudaMemcpy(img_d, img_h, height * width * channels * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(range_table_d, range_table, (QX_DEF_CHAR_MAX + 1) * sizeof(float), cudaMemcpyHostToDevice);
    // invoke kernel
    naiveKernel<<<num_blocks, threads_per_block>>>(
        img_d, img_tmp_d, map_factor_a_d, range_table_d,
        width, height, channel, sigma_spatial);
    // copy back img_tmp and map_factor
    cudaMemCpy(img_tmp_h, img_tmp_d, height * width * channels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemCpy(map_factor_a_h, map_factor_a_d, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(img_d);
    cudaFree(map_factor_a_d);
    // now we have img_tmp and map_factor_a on host, proceed to do the rest
}


___global___ void naiveKernel(
    unsigned char* img, float* img_temp, float* map_factor_a, 
    float* range_table, int width, int height, int channel, 
    float sigma_spatial) 
{
    int row_number = (blockIdx.x * blockDim.x) + threadIdx.x;
    float* temp_x  = &img_temp[row_number * width * channel];
    unsigned char* in_x = &img[row_number * width * channel];
    unsigned char* texture_x = &img[row_number * width * channel];

    float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * width)));
    float ypr, ypg, ypb, ycr, ycg, ycb;
    float fp, fc;
    float inv_alpha_ = 1 - alpha;

    *temp_x++ = ypr = *in_x++; 
    *temp_x++ = ypg = *in_x++; 
    *temp_x++ = ypb = *in_x++;
    unsigned char tpr = *texture_x++; 
    unsigned char tpg = *texture_x++;
    unsigned char tpb = *texture_x++;
    float * temp_factor_x = &map_factor_a[row_number * width];
    *temp_factor_x++ = fp = 1;

    for (int x = 1; x < width; x++) {
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
    for (int x = width - 2; x >= 0; x--) 
    {
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