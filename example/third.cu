#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

/* Cuda parallelization of the third for loop in RBF
    verify correctness 
    Input:  img, img_out_f, img_tmp, map_factor_a,
            map_factor_b, range_table
    Output: img_out_f
*/

void hostThird(unsigned char*, float*, float*. int, int, int, float);
void cudaThird(unsigned char*, float*, float*, int, int, int, float);
__constant__ float* range_table_const;
__global__ void thirdKernel(unsigned char*, float*, float*, int, int, int, float);


void hostThird(
    unsigned char* img, float* buffer, float* range_table, 
    int width, int height, int channel, float sigma_spatial)
{
    int width_height = width * height, width_channel = width * channel;
    int width_height_channel = width * height * channel;
    float * img_out_f = buffer;
    float * img_temp = &img_out_f[width_height_channel];
    float * map_factor_a = &img_temp[width_height_channel];
    float * map_factor_b = &map_factor_a[width_height];

    float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * height)));
    int h1 = height - 1;
    float inv_alpha_ = 1 - alpha;
    float * ycy, * ypy, * xcy;
    unsigned char * tcy, * tpy;
    float*ycf, *ypf, *xcf;

    for (int x = 0; x < width; x++) {
        tpy = &img[x * 3 + h1 * width_channel];
        tcy = tpy - width_channel;
        xcy = &img_temp[x * 3 + (h1 - 1) * width_channel];
        float at_ypf = map_factor_a[h1 * width + x];

        float at_ypy_r = img_temp[h1 * width_channel + x * 3];
        float at_ypy_g = img_temp[h1 * width_channel + x * 3 + 1];
        float at_ypy_b = img_temp[h1 * width_channel + x * 3 + 2];

        float* out_ = &img_out_f[x * 3 + (h1 - 1) * width_channel];
        xcf = &map_factor_a[(h1 -1)* width + x];
        float* factor_ = &map_factor_b[x + (h1-1) * width];

        for (int y = h1 - 1; y >= 0; y--) {
            unsigned char dr = abs((*tcy++) - (*tpy++));
            unsigned char dg = abs((*tcy++) - (*tpy++));
            unsigned char db = abs((*tcy++) - (*tpy++));
            int range_dist = (((dr << 1) + dg + db) >> 2);
            float weight = range_table[range_dist];
            float alpha_ = weight*alpha;

            float fcc = inv_alpha_*(*xcf) + alpha_*(at_ypf);
            at_ypf = fcc;
            *factor_ = 0.5f * (*factor_ + fcc);

            float ycc_r = inv_alpha_*(*xcy++) + alpha_* at_ypy_r;
            at_ypy_r = ycc_r;
        
            *out_ = 0.5f * (*out_ + ycc_r) / (*factor_);
            *out_++;

            float ycc_g = inv_alpha_*(*xcy++) + alpha_* at_ypy_g;
            at_ypy_g = ycc_g;
            *out_ = 0.5f * (*out_ + ycc_g) / (*factor_);
            *out_++;

            float ycc_b = inv_alpha_*(*xcy++) + alpha_* at_ypy_b;
            at_ypy_b = ycc_b;
            *out_ = 0.5f * (*out_ + ycc_b) / (*factor_);
            //*out_++;

            tcy = tcy - 3 - width_channel;
            tpy = tpy - 3 - width_channel;
            out_ = out_ - 2 - width_channel;
            xcy = xcy - 3 - width_channel;
            factor_ = factor_ - width;
            xcf = xcf - width;
        }
    }
}


void cudaThird(
    unsigned char* img, float* buffer, float* range_table, 
    int width, int height, int channel, float sigma_spatial)
{
    float* buffer_d;
    int buffer_len = width * height * channel * 2 + width * height * 3;
    cudaMalloc((void**) &buffer_d, buffer_len * sizeof(float));
    if (!buffer_d)
    {
        printf("Prelim Kernel: Cuda malloc fail on buffer_d");
        exit(1);
    }

    // img_d
    unsigned char *img_d;
    cudaMalloc((void **)&img_d, height * width * channel * sizeof(char));
    if (!img_d)
    {
        printf("Prelim Kernel: Cuda malloc fail on img_d");
        cudaFree(buffer_d);
        exit(1);
    }

    // range table
    //float *range_table_d;
    cudaMalloc((void **)&range_table_const, 256 * sizeof(float));
    if (!range_table_const)
    {
        printf("Prelim Kernel: Cuda malloc fail on range_table_d");
        cudaFree(buffer_d);
        cudaFree(img_d);
        exit(1);
    }
    // memcpy: only for testing, not needed in the real code
    cudaMemcpy(img_d, img, height * width * channel * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(buffer_d, buffer, buffer_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(range_table_const, range_table, 256 * sizeof(float), cudaMemcpyHostToDevice);

    int cols_per_block = 4;
    int num_blocks = (width % cols_per_block == 0) ? width / cols_per_block : width / cols_per_block + 1;

    dim3 grid(num_blocks, 1, 1);
    dim3 block(cols_per_block, 1, 1);
    thirdKernel<<<grid, block>>>(img_d, buffer_d, range_table_const, width, height, channel, sigma_spatial);

    cudaMemcpy(buffer, buffer_d, width * height * channel * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(buffer_d);
    cudaFree(range_table_const);
    cudaFree(img_d);
}


__global__ void thirdKernel(
    unsigned char* img, float* buffer, float* range_table, 
    int width, int height, int channel, float sigma_spatial)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int width_height = width * height, width_channel = width * channel;
    int width_height_channel = width * height * channel;
    float * img_out_f = buffer;
    float * img_temp = &img_out_f[width_height_channel];
    float * map_factor_a = &img_temp[width_height_channel];
    float * map_factor_b = &map_factor_a[width_height];

    int h1 = height - 1;
    float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * height))), inv_alpha_ = 1 - alpha;
    float* xcy;
    unsigned char *tcy, *tpy;
    float *xcf;

    tpy = &img[x * 3 + h1 * width_channel];
    tcy = tpy - width_channel;
    xcy = &img_temp[x * 3 + (h1 - 1) * width_channel];
    float at_ypf = map_factor_a[h1 * width + x];

    float at_ypy_r = img_temp[h1 * width_channel + x * 3];
    float at_ypy_g = img_temp[h1 * width_channel + x * 3 + 1];
    float at_ypy_b = img_temp[h1 * width_channel + x * 3 + 2];

    float* out_ = &img_out_f[x * 3 + (h1 - 1) * width_channel];
    xcf = &map_factor_a[(h1 -1)* width + x];
    float* factor_ = &map_factor_b[x + (h1-1) * width];

    //float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * height)));

    for (int y = h1 - 1; y >= 0; y--) {
        unsigned char dr = abs((*tcy++) - (*tpy++));
        unsigned char dg = abs((*tcy++) - (*tpy++));
        unsigned char db = abs((*tcy++) - (*tpy++));
        int range_dist = (((dr << 1) + dg + db) >> 2);
        float weight = range_table[range_dist];
        float alpha_ = weight*alpha;

        float fcc = inv_alpha_*(*xcf) + alpha_*(at_ypf);
        at_ypf = fcc;
        *factor_ = 0.5f * (*factor_ + fcc);

        float ycc_r = inv_alpha_*(*xcy++) + alpha_* at_ypy_r;
        at_ypy_r = ycc_r;
    
        *out_ = 0.5f * (*out_ + ycc_r) / (*factor_);
        *out_++;

        float ycc_g = inv_alpha_*(*xcy++) + alpha_* at_ypy_g;
        at_ypy_g = ycc_g;
        *out_ = 0.5f * (*out_ + ycc_g) / (*factor_);
        *out_++;

        float ycc_b = inv_alpha_*(*xcy++) + alpha_* at_ypy_b;
        at_ypy_b = ycc_b;
        *out_ = 0.5f * (*out_ + ycc_b) / (*factor_);
        //*out_++;

        tcy = tcy - 3 - width_channel;
        tpy = tpy - 3 - width_channel;
        out_ = out_ - 2 - width_channel;
        xcy = xcy - 3 - width_channel;
        factor_ = factor_ - width;
        xcf = xcf - width;
    }
}


int main() {
    const int width = 4;
    const int height = 4;
    const int channel = 3;

    const float sigma_spatial = 0.5;
    const float sigma_range = 16;

    unsigned char img[] = {
      10,20,30, 20,30,40, 30,40,50, 40,50,60,
      10,20,30, 40,30,20, 30,40,50, 60,50,40,
      30,20,10, 20,30,40, 50,40,30, 40,50,60,
      20,10,30, 20,30,40, 50,30,50, 40,50,60
    };

    float* buffer = new float[128];
    for (int i = 0; i < 48; i++) {
        buffer[i] = exp(-sqrt(2.0 * i)) * 255;
    }
    for (int i = 48; i < 96; i++) {
        buffer[i] = exp(-sqrt(5.0 * i)) * 255;
    }
    for (int i = 96; i < 112; i++) {
        buffer[i] = 0.1f * i;
    }
    for (int i = 112; i < 128; i++) {
        buffer[i] = 0.15f * i;
    }

    float range_table[255];
    float inv_sigma_range = 1.0f / (sigma_range * 255);
    for (int i = 0; i <= 255; i++) 
        range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));
    
    float* out_f_host = new float[width * height * channel];

    hostThird(img, buffer, range_table, width, height, channel, sigma_spatial);
    memcpy(out_f_host, buffer, width * height * channel * sizeof(float));

    cudaThird(img, buffer, range_table, width, height, channel, sigma_spatial);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width * channel; j++) {
            if (out_f_host[i * width + j] != buffer[i * width + j]) {
                printf("not match at [%d][%d]: %d\n", i, j/3, j%3);
            }
        }
    }
    delete[] out_f_host;
}

