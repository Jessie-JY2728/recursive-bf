/* ----------------
Compare GPU and CPU version of third loop.
correctness and efficiency
---------------- */
#include <iostream>
#include <ctime>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <random>
#define QX_DEF_CHAR_MAX 255


class Timer {
private:
	unsigned long begTime;
public:
	void start() { begTime = clock(); }
	float elapsedTime() { return float((unsigned long)clock() - begTime) / CLOCKS_PER_SEC; }
};

__global__ void secondKernel (float *, unsigned char *, float*, int, int, int, float, float);

void HostSecond(float *, unsigned char *, float*, int, int, int, float, float);

void DeviceSecond(float *, unsigned char *, float*, int, int, int, float, float);




int main(int argc, char *argv[]){

/* ----create artificial input for testing the third loop. 
 ---1st used input
    img, img_out_f, in_factor, map_factor_b, range_table, in_factor,
    height, channel, width_channel, width, alpha, inv_alpha_
 ---2nd used input:
    buffer, sigma_spatial, sigma_range
 ---modified results:
    img_out_f, map_factor_b(for comparison)
-----*/

//creation of artificial input
    int width = 5760;
    int height = 2000;
    int channel = 3;
    int width_height_channel = width * height * channel;
    int width_channel = width * channel;
    int width_height = width * height;
    int buffer_size = (width_height_channel + width_height + width_channel + width) * 2;
    // float * buffer_ref = new float[buffer_size];
    float *buffer = new float[buffer_size];
    float *buffer2 = new float[buffer_size];

   //1. artificial buffer
    // Initialize the buffer with random float values between 0 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (size_t i = 0; i < buffer_size; ++i) {
        buffer[i] = distribution(gen);
        buffer2[i] = buffer[i];
        // buffer_ref[i] = buffer[i];
    }
    // for(int i = 0; i < buffer_size; i++){
    //     printf("current buffer[%d] = %f ", i, buffer[i]);
    // }

   //2. artificial image
    // Calculate the size of the image data
    size_t imgSize = static_cast<size_t>(width) * height * channel;

    // Allocate memory for the image data
    unsigned char* img = new unsigned char[imgSize];

    // Initialize the image data (for example, with random values)
    for (size_t i = 0; i < imgSize; ++i) {
        img[i] = rand() % 256; // Assign random values between 0 and 255
    }
    // for(int i = 0; i < imgSize; i++){
    //     printf("current img[%d] = %d ", i, img[i]);
    // }

    float sigma_spatial = 0.03;
    float sigma_range = 10;

   //3. rangetable
    float range_table[256];
    float inv_sigma_range = 1.0f / (sigma_range * 255);
    for (int i = 0; i <= 255; i++) 
        range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));
   //4. other
    float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * height)));
    float inv_alpha_ = 1 - alpha;


//////////////////// Test and Compare
    Timer T;
    T.start();
    HostSecond(buffer, img, range_table, width, height, channel, alpha, inv_alpha_);
    float seq_time = T.elapsedTime();

    T.start();
    DeviceSecond(buffer2, img, range_table, width, height, channel, alpha, inv_alpha_);
    float gpu_time = T.elapsedTime();

    printf("seq time is %.4f\n", seq_time);
    printf("gpu time is %.4f\n", gpu_time);

/////////////////// Correctness Comparison 
    // Use an epsilon value for tolerance
    const float epsilon = 1e-6;  // Adjust as needed

    for(int i = 0; i < buffer_size; i++){
       if(fabs(buffer[i] - buffer2[i]) > epsilon){
           printf("Results  are not correct, the orignal result is buffer[%d] = %f,"
           "whereas the refactored result buffer2[%d] = is %f\n", i, buffer[i], i, buffer2[i]);
       }
   }
//     for(int i = 0; i < buffer_size; i++){
//        if(buffer_ref[i] != buffer[i]){
//            printf("Results  are not correct, the orignal result is buffer_ref[%d] = %f,"
//            "whereas the refactored result buffer[%d] = is %f\n", i, buffer_ref[i], i, buffer[i]);
//        }
//    }
//     for(int i = 0; i < buffer_size; i++){
//        if(buffer_ref[i] != buffer2[i]){
//            printf("Results  are not correct, the orignal result is buffer_ref[%d] = %f,"
//            "whereas the refactored result buffer2[%d] = is %f\n", i, buffer_ref[i], i, buffer2[i]);
//        }
//    }
}

__global__ void secondKernel (
  
    float * buffer, unsigned char * img, float* range_table, int width, int height, int channel, float alpha
, float inv_alpha_){
    /*----
    there are a number of width tasks to parallize. (width is the number of pixels per row)
    the width might be large like 5760 or even larger, as there might be a limit of the
    number of threads per block. We might propose a (width/1024) 1D blocks(suppose 1024 threads/b)
    ---- */
        int width_channel = width * channel;
        int width_height_channel = width * height * channel;
        int width_height = width * height;
        float * img_out_f = buffer;
        float * img_temp = &img_out_f[width_height_channel];
        float * map_factor_a = &img_temp[width_height_channel];
        float * map_factor_b = &map_factor_a[width_height]; 
        float * in_factor = map_factor_a;

        float * ycy, * ypy, * xcy, * ycf, * ypf, * xcf;
        unsigned char *tcy, *tpy;

        int index =  blockIdx.x * blockDim.x + threadIdx.x;
        if(index  >= width) return;
        // printf("index is %d", index);
        tpy = &img[3 * index];
        tcy = &img[3 * index + width_channel];
        xcy = &img_temp[ 3 * index + width_channel];

        ypy = &img_out_f[3 * index];
        ycy = &img_out_f[3 * index + width_channel];

        xcf = &in_factor[index + width];
        ypf = &map_factor_b[index];
        ycf = &map_factor_b[index + width];

        for(int y = 1; y < height; y++){

            unsigned char dr = abs((*tcy++) - (*tpy++));
            unsigned char dg = abs((*tcy++) - (*tpy++));
            unsigned char db = abs((*tcy++) - (*tpy++));
            int range_dist = (((dr << 1) + dg + db) >> 2);
            float weight = range_table[range_dist];
            float alpha_ = weight*alpha;
            //pointer move across column direction
            for (int c = 0; c < channel; c++) 
                *ycy++ = inv_alpha_*(*xcy++) + alpha_*(*ypy++);
            *ycf++ = inv_alpha_*(*xcf++) + alpha_*(*ypf++);
            tpy = tpy - 3 + width_channel;
            tcy = tcy - 3 + width_channel;
            xcy = xcy - 3 + width_channel;

            ypy = ypy - 3 + width_channel;
            ycy = ycy - 3 + width_channel;

            xcf = xcf - 1 + width;
            ypf = ypf - 1 + width;
            ycf = ycf - 1 + width;
        }
}


void HostSecond(float * buffer, unsigned char * img, float * range_table, int width, int height, int channel, float alpha, float inv_alpha_){
    
    int width_channel = width * channel;
    int width_height_channel = width * height * channel;
    int width_height = width * height;
    float * img_out_f = buffer;
    float * img_temp = &img_out_f[width_height_channel];
    float * map_factor_a = &img_temp[width_height_channel];
    float * map_factor_b = &map_factor_a[width_height]; 
    float * in_factor = map_factor_a;

    float * ycy, * ypy, * xcy;
    unsigned char * tcy, * tpy;
    float*ycf, *ypf, *xcf;

    for (int y = 1; y < height; y++) 
    {
        tpy = &img[(y - 1) * width_channel];
        tcy = &img[y * width_channel];
        xcy = &img_temp[y * width_channel];
        ypy = &img_out_f[(y - 1) * width_channel];
        ycy = &img_out_f[y * width_channel];

        xcf = &in_factor[y * width];
        ypf = &map_factor_b[(y - 1) * width];
        ycf = &map_factor_b[y * width];
        for (int x = 0; x < width; x++)
        {
            unsigned char dr = abs((*tcy++) - (*tpy++));
            unsigned char dg = abs((*tcy++) - (*tpy++));
            unsigned char db = abs((*tcy++) - (*tpy++));
            int range_dist = (((dr << 1) + dg + db) >> 2);
            float weight = range_table[range_dist];
            float alpha_ = weight*alpha;
            for (int c = 0; c < channel; c++) 
                *ycy++ = inv_alpha_*(*xcy++) + alpha_*(*ypy++);
            *ycf++ = inv_alpha_*(*xcf++) + alpha_*(*ypf++);
        }
    }
}

void DeviceSecond(float * buffer, unsigned char * img, float * range_table, int width, int height, int channel, float alpha, float inv_alpha_){

    int width_channel = width * channel;
    int width_height_channel = width * height * channel;
    int width_height = width * height;
    int buffer_size = (width_height_channel + width_height + width_channel + width) * 2;

//allocate space for variables in the device
    float* buffer_d, * range_table_d;
    unsigned char * img_d;
    cudaMalloc((void**) &buffer_d, buffer_size * sizeof(float));
    if (!buffer_d)
    {
        printf("Prelim Kernel: Cuda malloc fail on buffer_d");
        exit(1);
    }
    cudaMalloc((void**) &range_table_d, (256) * sizeof(float));
    if (!range_table_d) {
        printf("Naive Kernel: Cuda malloc fail on range_table_d");
       // delete[] img_tmp_h;
       // delete[] map_factor_a_h;
        exit(1);
    }
    cudaMalloc((void**) &img_d, height * width * channel * sizeof(unsigned char));
    if (!img_d) {
        printf("Naive Kernel: Cuda malloc fail on img_d");
        cudaFree(img_d);
        //delete[] img_tmp_h;
        //delete[] map_factor_a_h;
        exit(1);
    }

// memcpy: 
    cudaMemcpy(img_d, img, height * width * channel * sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(buffer_d, buffer, buffer_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(range_table_d, range_table, 256 * sizeof(float), cudaMemcpyHostToDevice);

//call kernel function
    int num_blocks = (width%1024) == 0? (int)(width/1024) : (int)(width/1024) + 1;
    int threads_per_block = 1024;
    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    secondKernel<<<grid, block>>>(buffer_d, img_d, range_table_d, width, height, channel, alpha, inv_alpha_);
    
//copy back 
    cudaMemcpy(buffer, buffer_d, sizeof(float) * buffer_size, cudaMemcpyDeviceToHost);

//free
    cudaFree(buffer_d);
    cudaFree(img_d);
    cudaFree(range_table_d);

}


