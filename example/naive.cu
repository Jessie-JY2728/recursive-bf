#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include "stb_image_write.h"
#include "stb_image.h"

__global__ void firstkernel(float, float, int, int, int);
//__global__ void secondkernel(unsigned char *, float, float, int, int, int);


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
    float *map_factor_a_h;
    float *img_tmp_h;


    if (img == NULL) {
        printf("Error loading the image :(");
        exit(1);
    }
    printf("Loaded image: w=%d, h=%d, c=%d", width, height, channels);

    img_tmp_h = (float*)malloc(width * height * channels * sizeof(float));
    if (img_tmp_h == NULL) {
        printf("Cannot mallocate img_tmp_h\n");
        exit(1);
    }

    map_factor_a_h = (float*)malloc(width * height * sizeof(float));
    if (img_tmp_h == NULL) {
        printf("Cannot mallocate img_tmp_h\n");
        exit(1);
    }

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

    cudaMalloc((void**) &map_factor_a_d, height * width * sizeof(float));
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
    cudaMemCpy(img_tmp_h, img_tmp_d, height * width * channels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemCpy(map_factor_a_h, map_factor_a_d, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(img_d);
    cudaFree(map_factor_a_d);

    /*----------------------------------------*/
    /*    Unchanged second third loop         */
    /*----------------------------------------*/


    float *img_temp = img_tmp_h;
    float *map_factor_a = map_factor_a_h;
    float *img_out_f = (float*)malloc(width * height channel * sizeof(float));
    if (!img_out_f) exit(1);
    float *map_factor_b = (float*)malloc(width * height * sizeof(float));
    if (!map_factor_b) exit(1);
    float *slice_factor_a = (float*)malloc(width * channels * sizeof(float));
    if (!slice_factor_a) exit(1);
    float *slice_factor_b = (float*)malloc(width * channels * sizeof(float));
    if (!slice_factor_b) exit(1);
    float *line_factor_a = (float*)malloc(width * sizeof(float));
    if (!line_factor_a) exit(1);
    float *line_factor_b = (float*)malloc(width * sizeof(float));
    if (!line_factor_b) exit(1);

    alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * height)));
    inv_alpha_ = 1 - alpha;
    float * ycy, * ypy, * xcy;
    unsigned char * tcy, * tpy;
    memcpy(img_out_f, img_temp, sizeof(float)* width_channel);

    float * in_factor = map_factor_a;
    float*ycf, *ypf, *xcf;
    memcpy(map_factor_b, in_factor, sizeof(float) * width);

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

    int h1 = height - 1;
    ycf = line_factor_a;
    ypf = line_factor_b;
    memcpy(ypf, &in_factor[h1 * width], sizeof(float) * width);
    for (int x = 0; x < width; x++) 
        map_factor_b[h1 * width + x] = 0.5f*(map_factor_b[h1 * width + x] + ypf[x]);

    ycy = slice_factor_a;
    ypy = slice_factor_b;
    memcpy(ypy, &img_temp[h1 * width_channel], sizeof(float)* width_channel);
    int k = 0; 
    for (int x = 0; x < width; x++) {
        for (int c = 0; c < channel; c++) {
            int idx = (h1 * width + x) * channel + c;
            img_out_f[idx] = 0.5f*(img_out_f[idx] + ypy[k++]) / map_factor_b[h1 * width + x];
        }
    }

    for (int y = h1 - 1; y >= 0; y--)
    {
        tpy = &img[(y + 1) * width_channel];
        tcy = &img[y * width_channel];
        xcy = &img_temp[y * width_channel];
        float*ycy_ = ycy;
        float*ypy_ = ypy;
        float*out_ = &img_out_f[y * width_channel];

        xcf = &in_factor[y * width];
        float*ycf_ = ycf;
        float*ypf_ = ypf;
        float*factor_ = &map_factor_b[y * width];
        for (int x = 0; x < width; x++)
        {
            unsigned char dr = abs((*tcy++) - (*tpy++));
            unsigned char dg = abs((*tcy++) - (*tpy++));
            unsigned char db = abs((*tcy++) - (*tpy++));
            int range_dist = (((dr << 1) + dg + db) >> 2);
            float weight = range_table[range_dist];
            float alpha_ = weight*alpha;

            float fcc = inv_alpha_*(*xcf++) + alpha_*(*ypf_++);
            *ycf_++ = fcc;
            *factor_ = 0.5f * (*factor_ + fcc);

            for (int c = 0; c < channel; c++)
            {
                float ycc = inv_alpha_*(*xcy++) + alpha_*(*ypy_++);
                *ycy_++ = ycc;
                *out_ = 0.5f * (*out_ + ycc) / (*factor_);
                *out_++;
            }
            *factor_++;
        }
        memcpy(ypy, ycy, sizeof(float) * width_channel);
        memcpy(ypf, ycf, sizeof(float) * width);
    }

    for (int i = 0; i < width_height_channel; ++i)
        img[i] = static_cast<unsigned char>(img_out_f[i]);


    /*----------------------------------------*/
    /*     END of unchanged                   */
    /*----------------------------------------*/



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
    //cudaFree(img_d);

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