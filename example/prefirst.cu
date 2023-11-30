#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "stb_image_write.h"
#include "stb_image.h"
#define QX_DEF_CHAR_MAX 255

/*----------------------------------*/
/*    helper class timer            */
/*----------------------------------*/
class Timer {
private:
	unsigned long begTime;
public:
	void start() { begTime = clock(); }
	float elapsedTime() { return float((unsigned long)clock() - begTime) / CLOCKS_PER_SEC; }
};

__global__ void firstKernel(unsigned char*, float*, float*, int, int, int, float);
__global__ void naiveKernel(unsigned char*, float*, float*, int, int, int, float);
__global__ void prelimKernel(unsigned char*, float*, float*, int, int, int, float);
__constant__ float* range_table_d;

/*----------------------------------*/
/*    BEGIN CPU VERSION             */
/*----------------------------------*/

void hostVersion(
    unsigned char * img,
    float sigma_spatial, float sigma_range, 
    int width, int height, int channel,
    float * buffer = 0)
{
    const int width_height = width * height;
    const int width_channel = width * channel;
    const int width_height_channel = width * height * channel;

    bool is_buffer_internal = (buffer == 0);
    if (is_buffer_internal)
        buffer = new float[(width_height_channel + width_height 
                            + width_channel + width) * 2];

    float * img_out_f = buffer;
    float * img_temp = &img_out_f[width_height_channel];
    float * map_factor_a = &img_temp[width_height_channel];
    float * map_factor_b = &map_factor_a[width_height]; 
    float * slice_factor_a = &map_factor_b[width_height];
    float * slice_factor_b = &slice_factor_a[width_channel];
    float * line_factor_a = &slice_factor_b[width_channel];
    float * line_factor_b = &line_factor_a[width];
    
    //compute a lookup table
    float range_table[QX_DEF_CHAR_MAX + 1];
    float inv_sigma_range = 1.0f / (sigma_range * QX_DEF_CHAR_MAX);
    for (int i = 0; i <= QX_DEF_CHAR_MAX; i++) 
        range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));

    float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * width)));
    float ypr, ypg, ypb, ycr, ycg, ycb;
    float fp, fc;
    float inv_alpha_ = 1 - alpha;

    // REFACTOR: store alpha_ of each pixel in map_factor_b for l->r, map_factor_c for r->l, use wh threads
    float *map_factor_c = new float[width_height];
    for (int y = 0; y < height; y++) {
        unsigned char* texture_x = &img[y * width_channel];
        float* map_alphas_l = &map_factor_b[y * width];
        float* map_alphas_r = &map_factor_c[y * width];
        for (int x = 1; x < width; x++) {               
            unsigned char tpr = texture_x[x * 3 - 3], tcr = texture_x[x * 3];
            unsigned char tpg = texture_x[x * 3 - 2], tcg = texture_x[x * 3 + 1];
            unsigned char tpb = texture_x[x * 3 - 1], tcb = texture_x[x * 3 + 2];
            unsigned char dr = abs(tcr - tpr);
            unsigned char dg = abs(tcg - tpg);
            unsigned char db = abs(tcb - tpb);
            int range_dist = (((dr << 1) + dg + db) >> 2);
            float weight = range_table[range_dist];
            float alpha_ = weight*alpha;
            map_alphas_l[x] = alpha_;
            range_dist = (((db << 1) + dg + dr) >> 2);
            map_alphas_r[x] = range_table[range_dist] * alpha;
        }
    }


    // REFACTOR: left to right and right to left with pre computed alpha_, can use wc threads + another stream for temp_factor
    for (int y = 0; y < height; y++) {
        float * temp_x = &img_temp[y * width_channel];
        unsigned char * in_x = &img[y * width_channel];
        *temp_x++ = ypr = *in_x++; 
        *temp_x++ = ypg = *in_x++; 
        *temp_x++ = ypb = *in_x++;
        float* map_factor_alphas = &map_factor_b[y * width];
        float * temp_factor_x = &map_factor_a[y * width];
        *temp_factor_x++ = fp = 1;

        // left to right
        for (int x = 1; x < width; x++) {
            float alpha_ = map_factor_alphas[x];
            //printf("%d ", alpha_);
            *temp_x++ = ycr = inv_alpha_*(*in_x++) + alpha_*ypr; 
            *temp_x++ = ycg = inv_alpha_*(*in_x++) + alpha_*ypg; 
            *temp_x++ = ycb = inv_alpha_*(*in_x++) + alpha_*ypb;
            ypr = ycr; ypg = ycg; ypb = ycb;

            *temp_factor_x++ = fc = inv_alpha_ + alpha_*fp;
            fp = fc;
        }
        //printf("\n\n");
        *--temp_x; *temp_x = 0.5f*((*temp_x) + (*--in_x));
        *--temp_x; *temp_x = 0.5f*((*temp_x) + (*--in_x));
        *--temp_x; *temp_x = 0.5f*((*temp_x) + (*--in_x));
        ypr = *in_x; ypg = *in_x; ypb = *in_x;

        *--temp_factor_x; *temp_factor_x = 0.5f*((*temp_factor_x) + 1);
        fp = 1;
        map_factor_alphas = &map_factor_c[y * width];
        //right to left
        for (int x = width - 2; x >= 0; x--) {
            float alpha_ = map_factor_alphas[x+1];
            //printf("%d ", alpha_);
            ycr = inv_alpha_ * (*--in_x) + alpha_ * ypr; 
            ycg = inv_alpha_ * (*--in_x) + alpha_ * ypg; 
            ycb = inv_alpha_ * (*--in_x) + alpha_ * ypb;
            *--temp_x; *temp_x = 0.5f*((*temp_x) + ycr);
            *--temp_x; *temp_x = 0.5f*((*temp_x) + ycg);
            *--temp_x; *temp_x = 0.5f*((*temp_x) + ycb);
            ypr = ycr; ypg = ycg; ypb = ycb;

            fc = inv_alpha_ + alpha_*fp;
            *--temp_factor_x; 
            *temp_factor_x = 0.5f*((*temp_factor_x) + fc);
            fp = fc;
        }
        //printf("\n");
    }
    delete[] map_factor_c;
    if (is_buffer_internal) delete[] buffer;   
  }

/*----------------------------------*/
/*     END CPU VERSION              */
/*     BEGIN SEPARATE PRELIM        */
/*----------------------------------*/

void separatePrelim(
    unsigned char *img_h, int width, int height, int channel,
    float sigma_spatial, float sigma_range, int block_side)
{
  // device allocation
  // buffer_d
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
  cudaMalloc((void **)&range_table_d, (QX_DEF_CHAR_MAX + 1) * sizeof(float));
  if (!range_table_d)
  {
    printf("Prelim Kernel: Cuda malloc fail on range_table_d");
    cudaFree(buffer_d);
    cudaFree(img_d);
    exit(1);
  }

  // host allocation buffer_h
  float* buffer_h = new float[buffer_len];
  float* img_tmp_h = &buffer_h[width * height * channel];
  float* map_factor_a_h = &img_tmp_h[width * height * channel];
  float* map_factor_b_h = &map_factor_a_h[width * height];
  float* map_factor_c_h = &map_factor_b_h[width * height];

  // range table
  float range_table[QX_DEF_CHAR_MAX + 1];
  float inv_sigma_range = 1.0f / (sigma_range * QX_DEF_CHAR_MAX);
  for (int i = 0; i <= QX_DEF_CHAR_MAX; i++)
    range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));

  cudaMemcpy(img_d, img_h, height * width * channel * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(range_table_d, range_table, (QX_DEF_CHAR_MAX + 1) * sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid(16, 16, 1);
  dim3 block(block_side, block_side, 1);

  prelimKernel<<<grid, block>>>(img_d, buffer_d, range_table_d, width, height, channel, sigma_spatial);

  cudaDeviceSynchronize();

  int num_blocks = (height % block_side) == 0 ? height / block_side : height / block_side + 1;
  dim3 grid_first(num_blocks, 1, 1);
  dim3 block_first(block_side, 1, 1);

  firstKernel<<<grid_first, block_first>>>(img_d, range_table_d, buffer_d, width, height, channel, sigma_spatial);
  float* img_tmp_d = &buffer_d[width * height * channel];
  float* map_factor_a_d = &img_tmp_d[width * height * channel];
  //cudaMemcpy(img_tmp_h, img_tmp_d, width * height * channel * sizeof(float), cudaMemcpyDeviceToHost);
  //cudaMemcpy(map_factor_a_h, map_factor_a_d, width * height * sizeof (float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  delete[] buffer_h;
  cudaFree(buffer_d);
}

__global__ void prelimKernel(
    unsigned char *img, float* buffer,
    float *range_table, int width, int height, int channel,
    float sigma_spatial)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    float* map_factor_b = &buffer[width * height * channel * 2 + width * height];
    float* map_factor_c = &map_factor_b[width * height];

    unsigned int row_step = blockDim.x * gridDim.x;
    unsigned int col_step = blockDim.y * gridDim.y;
    float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * width)));

    for (unsigned int r = row; r < height; r += row_step) {
        for (unsigned int c = col; c < width; c += col_step) {
            int index = r * width + c;
            if (c == 0) {
                map_factor_b[index] = 0;
                map_factor_c[index] = 0;
            } else {
                unsigned char tpr = img[index * 3 - 3], tcr = img[index * 3];
                unsigned char tpg = img[index * 3 - 2], tcg = img[index * 3 + 1];
                unsigned char tpb = img[index * 3 - 1], tcb = img[index * 3 + 2];
                unsigned char dr = abs(tcr - tpr);
                unsigned char dg = abs(tcg - tpg);
                unsigned char db = abs(tcb - tpb);

                int range_dist = (((dr << 1) + dg + db) >> 2);
                map_factor_b[index] = alpha * range_table[range_dist];

                range_dist = (((db << 1) + dg + dr) >> 2);
                map_factor_c[index] = alpha * range_table[range_dist];
            }
        }
        __syncthreads();
    }
}

__global__ void firstKernel(
    unsigned char* img, float* range_table, float* buffer, 
    int width, int height, int channel, float sigma_spatial) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * width)));
    float ypr, ypg, ypb, ycr, ycg, ycb;
    float fp, fc;
    float inv_alpha_ = 1 - alpha;

    // get to the parts of buffer
    float* img_tmp = &buffer[width * height * channel];
    float* map_factor_a = &img_tmp[width * height * channel];
    float* map_factor_b = &map_factor_a[width * height];
    float* map_factor_c = &map_factor_b[width * height];
    // get to the row
    map_factor_b += row * width;
    map_factor_c += row * width;

    float* temp_factor_x = &map_factor_a[row * width];
    float* temp_x = img_tmp + width * channel * row;
    unsigned char* in_x = &img[width * channel * row];
    *temp_factor_x++ = fp = 1;

    *temp_x++ = ypr = *in_x++; 
    *temp_x++ = ypg = *in_x++; 
    *temp_x++ = ypb = *in_x++;

    for (int x = 1; x < width; x++) {
        float alpha_ = map_factor_b[x];
        *temp_x++ = ycr = inv_alpha_*(*in_x++) + alpha_*ypr; 
        *temp_x++ = ycg = inv_alpha_*(*in_x++) + alpha_*ypg; 
        *temp_x++ = ycb = inv_alpha_*(*in_x++) + alpha_*ypb;
        ypr = ycr; ypg = ycg; ypb = ycb;

        *temp_factor_x++ = fc = inv_alpha_ + alpha_*fp;
        fp = fc;

    }
    *--temp_x; *temp_x = 0.5f*((*temp_x) + (*--in_x));
    *--temp_x; *temp_x = 0.5f*((*temp_x) + (*--in_x));
    *--temp_x; *temp_x = 0.5f*((*temp_x) + (*--in_x));
    ypr = *in_x; ypg = *in_x; ypb = *in_x;

    *--temp_factor_x; *temp_factor_x = 0.5f*((*temp_factor_x) + 1);
    fp = 1;
    
    for (int x = width - 2; x >= 0; x--) {
        float alpha_ = map_factor_c[x + 1];
        ycr = inv_alpha_ * (*--in_x) + alpha_ * ypr; 
        ycg = inv_alpha_ * (*--in_x) + alpha_ * ypg; 
        ycb = inv_alpha_ * (*--in_x) + alpha_ * ypb;
        *--temp_x; *temp_x = 0.5f*((*temp_x) + ycr);
        *--temp_x; *temp_x = 0.5f*((*temp_x) + ycg);
        *--temp_x; *temp_x = 0.5f*((*temp_x) + ycb);
        ypr = ycr; ypg = ycg; ypb = ycb;

        fc = inv_alpha_ + alpha_*fp;
        *--temp_factor_x; 
        *temp_factor_x = 0.5f*((*temp_factor_x) + fc);
        fp = fc;
    }  
}

/*----------------------------------*/
/*    END SEPARATE PRELIM           */
/*    BEGIN NAIVE FIRST             */
/*----------------------------------*/

void naiveFirst(
    unsigned char *img_h, int width, int height, int channel,
    float sigma_spatial, float sigma_range, int block_side)
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
  cudaMalloc((void **)&range_table_d, (QX_DEF_CHAR_MAX + 1) * sizeof(float));
  if (!range_table_d)
  {
    printf("Prelim Kernel: Cuda malloc fail on range_table_d");
    cudaFree(buffer_d);
    cudaFree(img_d);
    exit(1);
  }

  // host allocation buffer_h
  float* buffer_h = new float[buffer_len];
  float* img_tmp_h = &buffer_h[width * height * channel];
  float* map_factor_a_h = &img_tmp_h[width * height * channel];
  float* map_factor_b_h = &map_factor_a_h[width * height];
  float* map_factor_c_h = &map_factor_b_h[width * height];

  // range table
  float range_table[QX_DEF_CHAR_MAX + 1];
  float inv_sigma_range = 1.0f / (sigma_range * QX_DEF_CHAR_MAX);
  for (int i = 0; i <= QX_DEF_CHAR_MAX; i++)
    range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));

  cudaMemcpy(img_d, img_h, height * width * channel * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(range_table_d, range_table, (QX_DEF_CHAR_MAX + 1) * sizeof(float), cudaMemcpyHostToDevice);

  int num_blocks = (height % block_side) == 0 ? height / block_side : height / block_side + 1;
  dim3 grid_first(num_blocks, 1, 1);
  dim3 block_first(block_side, 1, 1);
  naiveKernel<<<grid_first, block_first>>>(img_d, range_table_d, buffer_d, width, height, channel, sigma_spatial);
  float* img_tmp_d = &buffer_d[width * height * channel];
  float* map_factor_a_d = &img_tmp_d[width * height * channel];
  cudaDeviceSynchronize();

  delete[] buffer_h;
  cudaFree(buffer_d);

}

__global__ void naiveKernel(
    unsigned char* img, float* range_table, float* buffer, 
    int width, int height, int channel, float sigma_spatial) 
{
    int row_number = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row_number >= height) return;   // row index out of bound
    float* img_tmp = &buffer[width * height * channel];
    float* map_factor_a = &img_tmp[width * height * channel];
    float* map_factor_b = &map_factor_a[width * height];
    float* map_factor_c = &map_factor_b[width * height];

    float* temp_x  = &img_tmp[row_number * width * channel];
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

/*----------------------------------*/
/*    END Naive First               */
/*----------------------------------*/


int main(int argc, char* argv[])
{
    if (argc != 3) {
        printf("usage: ./pf <in_file> <rows_per_block>");
        exit(1);
    }
    const char *filename_in = argv[1];
    const int rows_per_block = atoi(argv[2]);
    const float sigma_spatial = 32;
    const float sigma_range = 32;

    int width, height, channel;
    unsigned char *image = stbi_load(filename_in, &width, &height, &channel, 0);
    if (!image) {
        printf("Low Rating stb has FAILED to load Input Image. SAD.");
        exit(1);
    }
    printf("Loaded image: w=%d, h=%d, c=%d\n", width, height, channel);    
    Timer timer;
    timer.start();
    separatePrelim(image, width, height, channel, sigma_spatial, sigma_range, rows_per_block);
    float septime = timer.elapsedTime();

    timer.start();
    naiveFirst(image, width, height, channel, sigma_spatial, sigma_range, rows_per_block);
    float navtime = timer.elapsedTime();

    timer.start();
    hostVersion(image, sigma_spatial, sigma_range, width, height, channel);
    float hostime = timer.elapsedTime();
    printf("sep time: %.4f\n", septime);
    printf("nav time: %.4f\n", navtime);
    printf("cpu time: %.4f\n", hostime);
    return 0;
}
