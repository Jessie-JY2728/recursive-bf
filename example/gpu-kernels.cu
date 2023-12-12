#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#define QX_DEF_CHAR_MAX 255

__global__ void naiveKernel(unsigned char*, float*, float*, float* , int, int, int, float);
__global__ void firstKernel(unsigned char*, float*, float*,  int, int, int, float);
__global__ void secondKernel(unsigned char*, float*, float*,  int, int, int, float);
__global__ void lastKernel(unsigned char*, float*, int, int);
__constant__ float *range_table_const;

void naiveRemainder(unsigned char*, float*, float*, int, int, int, float, float);

/*--------------------------------*/
/*   Naive Section                */
/*--------------------------------*/

void invokeNaiveKernel(
    unsigned char* img_h, int width, int height, int channel,
    float sigma_spatial, float sigma_range, int rows_per_block, float* buffer
    ) 
{
    unsigned char* img_d;   // image on device
    float *img_tmp_d;   // img_temp, on device
    float *map_factor_a_d;  // map_factor_a, on device
    float *range_table_d;   // range table, on device
    
    int width_height_channel = width * height * channel;
    int width_height = width * height;
    int width_channel = width * channel;
    
    float * img_out_f = buffer;
    float * img_tmp_h = &img_out_f[width_height_channel];
    float * map_factor_a_h = &img_tmp_h[width_height_channel];
    float * map_factor_b = &map_factor_a_h[width_height]; 
    float * slice_factor_a = &map_factor_b[width_height];
    float * slice_factor_b = &slice_factor_a[width_channel];
    float * line_factor_a = &slice_factor_b[width_channel];
    float * line_factor_b = &line_factor_a[width];

    // range table for look up
    float range_table[QX_DEF_CHAR_MAX + 1];
    float inv_sigma_range = 1.0f / (sigma_range * QX_DEF_CHAR_MAX);
    for (int i = 0; i <= QX_DEF_CHAR_MAX; i++) 
        range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));

    // initialize on host side
    //img_tmp_h = new float[width * height * channel];
    //map_factor_a_h = new float[width * height];

    // initialize on device using cudaMalloc
    cudaMalloc((void**) &range_table_d, (QX_DEF_CHAR_MAX + 1) * sizeof(float));
    if (!range_table_d) {
        printf("Naive Kernel: Cuda malloc fail on range_table_d");
       // delete[] img_tmp_h;
       // delete[] map_factor_a_h;
        exit(1);
    }
    cudaMalloc((void**) &img_d, height * width * channel * sizeof(char));
    if (!img_d) {
        printf("Naive Kernel: Cuda malloc fail on img_d");
        cudaFree(range_table_d);
        //delete[] img_tmp_h;
        //delete[] map_factor_a_h;
        exit(1);
    }

    cudaMalloc((void**) &img_tmp_d, height * width * channel * sizeof(float));
    if (!img_tmp_d) {
        printf("Naive Kernel: Cuda malloc fail on img_tmp_d");
        //delete[] img_tmp_h;
        //delete[] map_factor_a_h;
        cudaFree(img_d);
        cudaFree(range_table_d);
        exit(1);
    }

    cudaMalloc((void**) &map_factor_a_d, height * width * sizeof(float));
    if (!map_factor_a_d) {
        printf("Naive Kernel: Cuda malloc fail on map_factor_a_d");
        //delete[] img_tmp_h;
        //delete[] map_factor_a_h;
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
    cudaMemcpy(img_d, img_h, height * width * channel * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(range_table_d, range_table, (QX_DEF_CHAR_MAX + 1) * sizeof(float), cudaMemcpyHostToDevice);
    // invoke kernel
    naiveKernel<<<num_blocks, threads_per_block>>>(
        img_d, img_tmp_d, map_factor_a_d, range_table_d,
        width, height, channel, sigma_spatial);
    // copy back img_tmp and map_factor
    cudaMemcpy(img_tmp_h, img_tmp_d, height * width * channel * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(map_factor_a_h, map_factor_a_d, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(img_d);
    cudaFree(map_factor_a_d);
    // now we have img_tmp and map_factor_a on host, proceed to do the rest

    naiveRemainder(img_h, buffer, range_table, width, height, channel, sigma_spatial, sigma_range);

}

/*----- Naive Kernel -----*/

__global__ void naiveKernel(
    unsigned char* img, float* img_temp, float* map_factor_a, 
    float* range_table, int width, int height, int channel, 
    float sigma_spatial) 
{
    int row_number = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row_number >= height) return;   // row index out of bound

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

/*----- Remaining CPU Computations -----*/
void naiveRemainder(
    unsigned char* img, float* buffer, float* range_table,
    int width, int height, int channel, float sigma_spatial, float sigma_range) 
{
    int width_channel = width * channel;
    int width_height = width * height;
    int width_height_channel = width_height * channel;
    float * img_out_f = buffer;
    float * img_temp = &img_out_f[width_height_channel];
    float * map_factor_a = &img_temp[width_height_channel];
    float * map_factor_b = &map_factor_a[width_height]; 
    float * slice_factor_a = &map_factor_b[width_height];
    float * slice_factor_b = &slice_factor_a[width_channel];
    float * line_factor_a = &slice_factor_b[width_channel];
    float * line_factor_b = &line_factor_a[width];

    float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * height)));
    float inv_alpha_ = 1 - alpha;
    float * ycy, * ypy, * xcy;
    unsigned char * tcy, * tpy;

    memcpy(img_out_f, img_temp, sizeof(float)* width_channel);

    float *in_factor = map_factor_a;
    float *ycf, *ypf, *xcf;
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

    for (int i = 0; i < height * width_channel; ++i){
	//printf("%.4f  ", img_out_f[i]);
	//if (i % 99 ==0) printf("\n");
        img[i] = static_cast<unsigned char>(img_out_f[i]);}
        //printf("\n\n");
    
}

/*--------------------------------*/
/*   END Naive Section            */
/*   Begin Full Refactor          */
/*--------------------------------*/

void refactorGPU(
    unsigned char* img_h, int width, int height, int channel,
    float sigma_spatial, float sigma_range, int rows_per_block) 
{   
    // initial steps
    unsigned char* img_d;   // image on device
    float* buffer_d;    // img_out_f_d, img_tmp_d, map_factor_a_d, map_factor_b_d
    
    int width_height_channel = width * height * channel;
    int width_height = width * height;
    int width_channel = width * channel;
    
    // range table for look up
    float range_table[QX_DEF_CHAR_MAX + 1];
    float inv_sigma_range = 1.0f / (sigma_range * QX_DEF_CHAR_MAX);
    for (int i = 0; i <= QX_DEF_CHAR_MAX; i++) 
        range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));

    // copy range table to device
    cudaMalloc((void**) &range_table_const, (QX_DEF_CHAR_MAX + 1) * sizeof(float));
    if (!range_table_const) {
        //delete[] range_table;
        printf("cannot allocate range table on device\n");
        exit(1);
    }
    //cudaMemcpyToSymbol(range_table_const, range_table, QX_DEF_CHAR_MAX * sizeof(float));
    //delete[] range_table;
    cudaMemcpy(range_table_const, range_table, 256 * sizeof(float), cudaMemcpyHostToDevice);
    // float buffer_d[img_out_f_d, img_tmp_d, map_factor_a, map_factor_b]
    int buffer_d_len = width_height_channel * 2 + width_height * 2;
    cudaMalloc((void**) &buffer_d, buffer_d_len * sizeof(float));
    if (!buffer_d) {
        cudaFree(range_table_const);
        printf("cannot allocate buffer on device\n");
        exit(1);
    }

    // copy input image to device
    cudaMalloc((void**) &img_d, height * width * channel * sizeof(char));
    if (!img_d) {
        cudaFree(range_table_const);
        cudaFree(buffer_d);
        exit(1);
    }
    cudaMemcpy(img_d, img_h, height * width * channel * sizeof(char), cudaMemcpyHostToDevice);

    // first kernel: like first for loop in original cpu code,
    // process image left to right then right to left, each row independent
    // computes img_temp and map_factor_a
    int num_blocks = (height % rows_per_block) == 0 ? height / rows_per_block : height / rows_per_block + 1;
    dim3 grid_first(num_blocks, 1, 1);
    dim3 block_first(rows_per_block, 1, 1);
    float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * width)));
    firstKernel<<<grid_first, block_first>>>(img_d, range_table_const, buffer_d, width, height, channel, alpha);
    // no need to copy anything back, sync and begin second kernel
    cudaDeviceSynchronize();

    // second kernel: like second and third for loop in original cpu code
    // process image top to bottom, computing img_out_f and map_factor_b
    // then bottom to top, getting the final img_out_f
    alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * height)));
    num_blocks = (width % rows_per_block == 0) ? width / rows_per_block : width / rows_per_block + 1;
    dim3 grid_second(num_blocks, 1, 1);
    dim3 block_second(rows_per_block, 1, 1);
    secondKernel<<<grid_second, block_second>>>(img_d, range_table_const, buffer_d, width, height, channel, alpha);
    cudaDeviceSynchronize();

    // last kernel: convert to unsigned char
    lastKernel<<<grid_first, block_first>>>(img_d, buffer_d, width_channel, height);
    cudaMemcpy(img_h, img_d, width_height_channel * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(buffer_d);
    cudaFree(img_d);
    cudaFree(range_table_const);
}

// input: img; output: img_tmp, map_factor_a
__global__ void firstKernel(
    unsigned char* img, float* range_table, float* buffer, 
    int width, int height, int channel, float alpha) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    //float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * width)));
    float ypr, ypg, ypb, ycr, ycg, ycb;
    float fp, fc;
    float inv_alpha_ = 1 - alpha;

    // get to the parts of buffer
    float* img_tmp = &buffer[width * height * channel];
    float* map_factor_a = &img_tmp[width * height * channel];
    
    // get to the row
    float * temp_factor_x = &map_factor_a[row * width];
    float* temp_x  = &img_tmp[row * width * channel];
    unsigned char* in_x = &img[row * width * channel];
    unsigned char* texture_x = &img[row * width * channel];

    unsigned char tpr = *texture_x++; 
    unsigned char tpg = *texture_x++;
    unsigned char tpb = *texture_x++;
    *temp_factor_x++ = fp = 1;

    *temp_x++ = ypr = *in_x++; 
    *temp_x++ = ypg = *in_x++; 
    *temp_x++ = ypb = *in_x++;

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


__global__ void secondKernel(
    unsigned char* img, float* range_table, float* buffer, 
    int width, int height, int channel, float alpha)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width) return;

    float inv_alpha_ = 1 - alpha;
    int width_channel = width * channel;
    int width_height_channel = width * height * channel;
    int width_height = width * height;

    float * ycy, * ypy, * xcy, * ycf, * ypf, * xcf;
    unsigned char *tcy, *tpy;

    float * img_out_f = buffer;
    float * img_temp = &img_out_f[width_height_channel];
    float * map_factor_a = &img_temp[width_height_channel];
    float * map_factor_b = &map_factor_a[width_height]; 
    float * in_factor = map_factor_a;
    tpy = &img[3 * index];
    tcy = &img[3 * index + width_channel];
    xcy = &img_temp[3 * index + width_channel];

    img_out_f[3 * index] = img_temp[3 * index]; 
    img_out_f[3 * index + 1] = img_temp[3 * index + 1]; 
    img_out_f[3 * index + 2] = img_temp[3 * index + 2];
    map_factor_b[index] = map_factor_a[index];

    ypy = &img_out_f[3 * index];
    ycy = &img_out_f[3 * index + width_channel];

    xcf = &in_factor[index + width];
    ypf = &map_factor_b[index];
    ycf = &map_factor_b[index + width];

    // top tp bottom
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

    int h1 = height - 1, x = index;

    //in_factor = &map_factor_a[h1 * width];
    //ypf = &map_factor_b[h1 * width];
    //ypf[index] = 0.5f*(ypf[index] + in_factor[index]);
    map_factor_b[h1 * width + x] = 0.5f * (map_factor_b[h1 * width + x] + map_factor_a[h1 * width + x]); 
    xcy = &img_temp[h1 * width_channel];
    ypy = &img_out_f[h1 * width_channel];

    ypy[index * 3] = 0.5f * (ypy[index * 3] + xcy[index * 3]) / map_factor_b[h1 * width + x];
    ypy[index * 3 + 1] = 0.5f * (ypy[index * 3 + 1] + xcy[index * 3 + 1]) / map_factor_b[h1 * width + x];
    ypy[index * 3 + 2] = 0.5f * (ypy[index * 3 + 2] + xcy[index * 3 + 2]) / map_factor_b[h1 * width + x];

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

    // bottom to top
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

        float ycc_b = inv_alpha_*(*xcy) + alpha_* at_ypy_b;
        at_ypy_b = ycc_b;
        *out_ = 0.5f * (*out_ + ycc_b) / (*factor_);

        tcy = tcy - 3 - width_channel;
        tpy = tpy - 3 - width_channel;
        out_ = out_ - 2 - width_channel;
        xcy = xcy - 2 - width_channel;
        factor_ = factor_ - width;
        xcf = xcf - width;
    }
}


__global__ void lastKernel(unsigned char* img, float* img_out_f, int width_channel, int height)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    //int width_channel = width * channel;
    int start = row * width_channel;
    for (int i = 0; i < width_channel; i++) {
        img[start + i] = static_cast<unsigned char>(img_out_f[start + i]);

    }
}
