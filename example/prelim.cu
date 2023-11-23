#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#define QX_DEF_CHAR_MAX 255

void invokePrelimHost(
    unsigned char *img, int width, int height, int channel,
    float sigma_spatial, float sigma_range)
{
  // REFACTOR: store alpha_ of each pixel in map_factor_b for l->r, map_factor_c for r->l, use wh threads

  float *map_factor_b = new float[width * height];
  float *map_factor_c = new float[width * height];
  // range table
  float range_table[QX_DEF_CHAR_MAX + 1];
  float inv_sigma_range = 1.0f / (sigma_range * QX_DEF_CHAR_MAX);
  for (int i = 0; i <= QX_DEF_CHAR_MAX; i++)
    range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));
  float alpha = static_cast<float>(exp(-sqrt(2.0) / (sigma_spatial * width)));

  for (int y = 0; y < height; y++)
  {
    unsigned char *texture_x = &img[y * width * channel];
    float *map_alphas_l = &map_factor_b[y * width];
    float *map_alphas_r = &map_factor_c[y * width];
    for (int x = 1; x < width; x++)
    {
      unsigned char tpr = texture_x[x * 3 - 3], tcr = texture_x[x * 3];
      unsigned char tpg = texture_x[x * 3 - 2], tcg = texture_x[x * 3 + 1];
      unsigned char tpb = texture_x[x * 3 - 1], tcb = texture_x[x * 3 + 2];
      unsigned char dr = abs(tcr - tpr);
      unsigned char dg = abs(tcg - tpg);
      unsigned char db = abs(tcb - tpb);
      int range_dist = (((dr << 1) + dg + db) >> 2);
      float weight = range_table[range_dist];
      float alpha_ = weight * alpha;
      map_alphas_l[x] = alpha_;
      range_dist = (((db << 1) + dg + dr) >> 2);
      map_alphas_r[x] = range_table[range_dist] * alpha;
    }
  }

  // Print the values of map_factor_b_h
  printf("Values of map_factor_b:\n");
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      printf("%.4f ", map_factor_b[i * width + j]);
    }
    printf("\n");
  }

  // Print the values of map_factor_c_h
  printf("\nValues of map_factor_c:\n");
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      printf("%.4f ", map_factor_c[i * width + j]);
    }
    printf("\n");
  }
}

__global__ void prelimKernel(
    unsigned char *img, float *map_factor_b, float *map_factor_c,
    float *range_table, int width, int height, int channel,
    float sigma_spatial);

void invokePrelimKernel(
    unsigned char *img_h, int width, int height, int channel,
    float sigma_spatial, float sigma_range, int block_side)
{
  // device allocation
  // map_factor_b
  float *map_factor_b_d;
  cudaMalloc((void **)&map_factor_b_d, width * height * sizeof(float));
  if (!map_factor_b_d)
  {
    printf("Prelim Kernel: Cuda malloc fail on map_factor_b_d");
    exit(1);
  }

  // map_factor_c
  float *map_factor_c_d;
  cudaMalloc((void **)&map_factor_c_d, width * height * sizeof(float));
  if (!map_factor_c_d)
  {
    printf("Prelim Kernel: Cuda malloc fail on map_factor_c_d");
    cudaFree(map_factor_b_d);
    exit(1);
  }

  // img_d
  unsigned char *img_d;
  cudaMalloc((void **)&img_d, height * width * channel * sizeof(char));
  if (!img_d)
  {
    printf("Prelim Kernel: Cuda malloc fail on img_d");
    cudaFree(map_factor_b_d);
    cudaFree(map_factor_c_d);
    exit(1);
  }

  // range table
  float *range_table_d;
  cudaMalloc((void **)&range_table_d, (QX_DEF_CHAR_MAX + 1) * sizeof(float));
  if (!range_table_d)
  {
    printf("Prelim Kernel: Cuda malloc fail on range_table_d");
    cudaFree(map_factor_b_d);
    cudaFree(map_factor_c_d);
    cudaFree(img_d);
    exit(1);
  }

  // host allocation
  // map_factor_b
  float *map_factor_b_h = new float[width * height];

  // map_factor_c
  float *map_factor_c_h = new float[width * height];

  // range table
  float range_table[QX_DEF_CHAR_MAX + 1];
  float inv_sigma_range = 1.0f / (sigma_range * QX_DEF_CHAR_MAX);
  for (int i = 0; i <= QX_DEF_CHAR_MAX; i++)
    range_table[i] = static_cast<float>(exp(-i * inv_sigma_range));

  cudaMemcpy(img_d, img_h, height * width * channel * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(range_table_d, range_table, (QX_DEF_CHAR_MAX + 1) * sizeof(float), cudaMemcpyHostToDevice);

  int grid_rows = height % block_side == 0 ? height / block_side : height / block_side + 1;
  int grid_cols = width % block_side == 0 ? width / block_side : width / block_side + 1;

  dim3 grid(grid_rows, grid_cols, 1);
  dim3 block(block_side, block_side, 1);

  prelimKernel<<<grid, block>>>(
      img_d, map_factor_b_d, map_factor_c_d, range_table_d,
      width, height, channel, sigma_spatial);

  cudaDeviceSynchronize();
  cudaMemcpy(map_factor_b_h, map_factor_b_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(map_factor_c_h, map_factor_c_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);

  // Print the values of map_factor_b_h
  printf("Values of map_factor_b_h:\n");
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      printf("%.4f ", map_factor_b_h[i * width + j]);
    }
    printf("\n");
  }

  // Print the values of map_factor_c_h
  printf("\nValues of map_factor_c_h:\n");
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      printf("%.4f ", map_factor_c_h[i * width + j]);
    }
    printf("\n");
  }
  printf("\n\n");
}

__global__ void prelimKernel(
    unsigned char *img, float *map_factor_b, float *map_factor_c,
    float *range_table, int width, int height, int channel,
    float sigma_spatial)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

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

int main()
{
  // Input image data
  unsigned char img_h[] = {
      10,20,30, 20,30,40, 30,40,50, 40,50,60,
      10,20,30, 40,30,20, 30,40,50, 60,50,40,
      30,20,10, 20,30,40, 50,40,30, 40,50,60,
      20,10,30, 20,30,40, 50,30,50, 40,50,60
  };

  int width = 4;
  int height = 4;
  int channels = 3; // Grayscale image

  float sigma_spatial = 100;
  float sigma_range = 100;
  int rows_per_block = 2;

  invokePrelimKernel(img_h, width, height, channels, sigma_spatial, sigma_range, rows_per_block);
  invokePrelimHost(img_h, width, height, channels, sigma_spatial, sigma_range);

  return 0;
}