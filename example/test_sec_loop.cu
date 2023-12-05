/* ----------------
Compare GPU and CPU version of third loop.
correctness and efficiency
---------------- */
#include <iostream>
#include <ctime>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <random>
#define QX_DEF_CHAR_MAX 255

__global__ void secondKernel (
  
    unsigned char * img, float * img_temp, float * img_out_f, float * in_factor, float * map_factor_b,
    float* range_table, int width, int height, int channel, int width_channel, float alpha, float inv_alpha_);

class Timer {
private:
	unsigned long begTime;
public:
	void start() { begTime = clock(); }
	float elapsedTime() { return float((unsigned long)clock() - begTime) / CLOCKS_PER_SEC; }
};


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
    int width = 10;
    int height = 2;
    int channel = 3;
    int width_channel = width * channel;
    int width_height_channel = width * height * channel;
    int width_height = width * height;

    int buffer_size = (width_height_channel + width_height + width_channel + width) * 2;
    float *buffer = new float[buffer_size];

   //1. artificial buffer
    // Initialize the buffer with random float values between 0 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (size_t i = 0; i < buffer_size; ++i) {
        buffer[i] = distribution(gen);
    }
    // for(int i = 0; i < buffer_size; i++){
    //     printf("current buffer[%d] = %f ", i, buffer[i]);
    // }


   //2. artificial image
    // Calculate the size of the image data
    size_t imgSize = static_cast<size_t>(width) * height * channel;

    // Allocate memory for the image data
    unsigned char* img_h = new unsigned char[imgSize];

    // Initialize the image data (for example, with random values)
    for (size_t i = 0; i < imgSize; ++i) {
        img_h[i] = rand() % 256; // Assign random values between 0 and 255
    }
    // for(int i = 0; i < imgSize; i++){
    //     printf("current img_h[%d] = %d ", i, img_h[i]);
    // }


    float * img_out_f = buffer;
    float * img_temp = &img_out_f[width_height_channel];
    float * map_factor_a = &img_temp[width_height_channel];
    float * map_factor_b = &map_factor_a[width_height]; 
    float * slice_factor_a = &map_factor_b[width_height];
    float * slice_factor_b = &slice_factor_a[width_channel];
    float * line_factor_a = &slice_factor_b[width_channel];
    float * line_factor_b = &line_factor_a[width];


    float * in_factor = map_factor_a;

    float sigma_spatial = 0.1;
    float sigma_range = 0.03;

    //rangetable
    float range_table[QX_DEF_CHAR_MAX + 1];
    float inv_sigma_range = 1.0f / (sigma_range * QX_DEF_CHAR_MAX);
    for (int i = 0; i <= QX_DEF_CHAR_MAX; i++) 
        range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));
    
    float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * height)));
    float inv_alpha_ = 1 - alpha;
    float * ycy, * ypy, * xcy;
    unsigned char * tcy, * tpy;
    float*ycf, *ypf, *xcf;
    
    memcpy(img_out_f, img_temp, sizeof(float)* width_channel);
    memcpy(map_factor_b, in_factor, sizeof(float) * width);
//device variable declaration
    unsigned char * img_d;
    float * range_table_d, * img_out_f_d, *img_temp_d, * map_factor_b_d;
    float * in_factor_d;

    //for testing correctness of the changed output
    float* img_out_f_copy = new float[width_height_channel];
    memcpy(img_out_f_copy, img_out_f, sizeof(float) * width_height_channel);
    // memcpy(img_out_f_copy, img_temp, sizeof(float)* width_channel);
    float* map_factor_b_copy = new float[width_height];
    memcpy(map_factor_b_copy, map_factor_b, sizeof(float) * width_height);
    // memcpy(map_factor_b_copy, in_factor, sizeof(float) * width);

    //testing cpu refactor
    float* img_out_f_2 = new float[width_height_channel];
    memcpy(img_out_f_2, img_out_f, sizeof(float) * width_height_channel);
    float* map_factor_b_2 = new float[width_height];
    memcpy(map_factor_b_2, map_factor_b, sizeof(float) * width_height);

    Timer timer;
    float elapse;

//CPU version
    timer.start();  // start timer
    for (int y = 1; y < height; y++) 
    {
        tpy = &img_h[(y - 1) * width_channel];
        tcy = &img_h[y * width_channel];
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

//     for (int y = 1; y < height; y++)
//      {
//          tpy = &img_h[(y - 1) * width_channel];
//          tcy = &img_h[y * width_channel];
//          xcy = &img_temp[y * width_channel];
//          ypy = &img_out_f_2[(y - 1) * width_channel];
//          ycy = &img_out_f_2[y * width_channel];

//          xcf = &in_factor[y * width];
//          ypf = &map_factor_b_2[(y - 1) * width];
//          ycf = &map_factor_b_2[y * width];

//          for (int x = width - 1; x >= 0; x--)
//          {
//              unsigned char dr = abs((*tcy++) - (*tpy++));
//              unsigned char dg = abs((*tcy++) - (*tpy++));
//              unsigned char db = abs((*tcy++) - (*tpy++));
//              int range_dist = (((dr << 1) + dg + db) >> 2);
//              float weight = range_table[range_dist];
//              float alpha_ = weight*alpha;
//              for (int c = 0; c < channel; c++)
//                  *ycy++ = inv_alpha_*(*xcy++) + alpha_*(*ypy++);
//              *ycf++ = inv_alpha_*(*xcf++) + alpha_*(*ypf++);
//          }
//      }

//     for(int i = 0; i < width_height_channel; i++){
//        if(img_out_f[i] != img_out_f_2[i]){
//            printf("Results not are  correct, the orignal result is img_out_f[%d] = %f,"
//            "whereas the refactored result img_out_f_2[%d] = is %f\n", i, img_out_f[i], i, img_out_f_2[i]);
//        }
//    }

//     for(int i = 0; i < width_height; i++){
//         if(map_factor_b[i] != map_factor_b_2[i]){
//             printf("Results not are  correct, the orignal result is map_factor_b[%d] = %f, "
//             "whereas the refactored result map_factor_b_2[%d] = is %f\n", i, map_factor_b[i], i, map_factor_b_2[i]);
//         }
//     }
    elapse = timer.elapsedTime(); // runtime
	printf("CPU External Buffer: %2.5fsecs\n", elapse); // print runtime


//GPU version

    timer.start();
  //create variables in the device
    cudaMalloc((void**) &range_table_d, (QX_DEF_CHAR_MAX + 1) * sizeof(float));
    if (!range_table_d) {
        printf("Naive Kernel: Cuda malloc fail on range_table_d");
       // delete[] img_tmp_h;
       // delete[] map_factor_a_h;
        exit(1);
    }
    cudaMalloc((void**) &img_d, height * width * channel * sizeof(unsigned char));
    if (!img_d) {
        printf("Naive Kernel: Cuda malloc fail on img_d");
        cudaFree(range_table_d);
        //delete[] img_tmp_h;
        //delete[] map_factor_a_h;
        exit(1);
    }

    cudaMalloc((void**) &img_temp_d, height * width * channel * sizeof(float));
    if (!img_temp_d) {
        printf("Naive Kernel: Cuda malloc fail on img_temp_d");
        //delete[] img_tmp_h;
        //delete[] map_factor_a_h;
        cudaFree(img_temp_d);
        exit(1);
    }

    cudaMalloc((void**) &map_factor_b_d, height * width * sizeof(float));
    if (!map_factor_b_d) {
        printf("Naive Kernel: Cuda malloc fail on map_factor_b_d");
        //delete[] img_tmp_h;
        //delete[] map_factor_a_h;
        // cudaFree(img_d);
        // cudaFree(img_tmp_d);
        // cudaFree(range_table_d);
        exit(1);
    } 

    cudaMalloc((void **) &img_out_f_d, width_height_channel * sizeof(float));
    if(!img_out_f_d){
        printf("Naive Kernel: Cuda malloc fail on img_out_f_d");
    }
    
    cudaMalloc((void **) &in_factor, width_height * sizeof(float));
    if(!in_factor){
        printf("Naive Kernel: Cuda malloc fail on in_factor");
    }

    // copy input  to device
    cudaMemcpy(img_d, img_h, height * width * channel * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(range_table_d, range_table, (QX_DEF_CHAR_MAX + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(img_temp_d, img_temp, height * width * channel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(map_factor_b_d, map_factor_b_copy, height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(img_out_f_d, img_out_f_copy, width_height_channel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(in_factor_d, in_factor, width_height * sizeof(float), cudaMemcpyHostToDevice);
    int num_blocks = (width%1024) == 0? (int)(width/1024) : (int)(width/1024) + 1;
    int threads_per_block = 1024;
    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);
    secondKernel<<<grid, block>>>(img_d, img_temp_d, img_out_f_d, in_factor_d,
    map_factor_b_d, range_table_d, width, height, channel, width_channel, alpha, inv_alpha_);
  
    //copy back modified data and compare
    cudaMemcpy(img_out_f_copy, img_out_f_d, sizeof(float) * width_height_channel, cudaMemcpyDeviceToHost);
    cudaMemcpy(map_factor_b_copy, map_factor_b_d, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(img_d);
    cudaFree(range_table_d);
    cudaFree(map_factor_b_d);
    cudaFree(img_out_f_d);
    cudaFree(in_factor_d);    
    cudaFree(img_temp_d);

    elapse = timer.elapsedTime();   // runtime
    printf("GPU Naive Kernel: %2.5fsecs\n", elapse); // print runtime

//correctness comparison 

    for(int i = 0; i < width_height_channel; i++){
       if(img_out_f[i] != img_out_f_copy[i]){
           printf("Results not are  correct, the orignal result is img_out_f[%d] = %f,"
           "whereas the refactored result img_out_f_copy[%d] = is %f\n", i, img_out_f[i], i, img_out_f_copy[i]);
       }
   }
    for(int i = 0; i < width_height; i++){
        if(map_factor_b[i] != map_factor_b_copy[i]){
            printf("Results not are  correct, the orignal result is map_factor_b[%d] = %f, "
            "whereas the refactored result map_factor_b_copy[%d] = is %f\n", i, map_factor_b[i], i, map_factor_b_copy[i]);
        }
    }
//time comparision
}

__global__ void secondKernel (
  
    unsigned char * img, float * img_temp, float * img_out_f, float * in_factor, float * map_factor_b,
    float* range_table, int width, int height, int channel, int width_channel , float alpha
, float inv_alpha_){
    /*----
    there are a number of width tasks to parallize. (width is the number of pixels per row)
    the width might be large like 5760 or even larger, as there might be a limit of the
    number of threads per block. We might propose a (width/1024) 1D blocks(suppose 1024 threads/b)
    ---- */
        int index =  blockIdx.x * blockDim.x + threadIdx.x;
    //initialize parameters
        float * ycy, * ypy, * xcy, * ycf, * ypf, * xcf;
        unsigned char *tcy, *tpy;
        
        if(index  >= width) return;
        printf("index is %d", index);
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
