#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "gpu-kernels.h"
#include "stb_image_write.h"
#include "stb_image.h"
#include "../include/rbf.hpp"

class Timer {
private:
	unsigned long begTime;
public:
	void start() { begTime = clock(); }
	float elapsedTime() { return float((unsigned long)clock() - begTime) / CLOCKS_PER_SEC; }
};



int main(int argc, char *argv[]) {
    // parse args
    if (argc != 5)
	{
		printf("Usage:\n");
		printf("--------------------------------------------------------------------\n\n");
		printf("rbf filename_in filename_out rows_per_block who\n");
        printf("Where rows_per_block is how many rows a block will process\n");
        printf("who: 0=CPU, 1=GPU Naive, 2=GPU Refactor\n");
		printf("--------------------------------------------------------------------\n");
		return(-1);
	}
    //const char *filename_out = argv[1];
	const char *filename_in = argv[1];
    const char *filename_out = argv[2];
    //std::string filename_out(filename_in);
    int ROWS_PER_BLOCK = atoi(argv[3]);
    int who = atoi(argv[4]);

	float sigma_spatial = 0.03;
	float sigma_range = 0.1;
    int width, height, channel;
    unsigned char *image = stbi_load(filename_in, &width, &height, &channel, 0);
    if (!image) {
        printf("Low Rating stb has FAILED to load Input Image. SAD.");
        exit(1);
    }
    printf("Loaded image: w=%d, h=%d, c=%d\n", width, height, channel);
    // char* image: input file

    int width_height = width * height;
    int width_height_channel = width_height * channel;
    int width_channel = width * channel;

    Timer timer;
    float elapse;

    // CPU version
    if (who == 0) {
        unsigned char * image_out = 0;
        float *buffer = new float[(width_height_channel + width_height + width_channel + width) * 2];
        timer.start();  // start timer
        recursive_bf(image, image_out, sigma_spatial, sigma_range, width, height, channel, buffer); // use original rbf
        elapse = timer.elapsedTime(); // runtime
        printf("CPU External Buffer: %2.5fsecs\n", elapse); // print runtime
        delete[] buffer;    // clean up
        /*for (int i = 0; i < width_height_channel; i++) {
	  printf("%d ", image[i]);
	  if (i % 99 == 98) printf("\n");
	}
	printf("\n\n");*/
	//std::string cpu_filename_out = "cpu_" + filename_out;   // add prefix "cpu_" for output file name
        stbi_write_jpg(filename_out, width, height, channel, image_out, 75);   // write out cpu image
    } else if (who == 1) {
        // GPU naive kernel
        float* buffer = new float[(width_height_channel + width_height + width_channel + width) * 2];
        timer.start();
        invokeNaiveKernel(image, width, height, channel, sigma_spatial, sigma_range, ROWS_PER_BLOCK, buffer);
        elapse = timer.elapsedTime();   // runtime
        printf("GPU Naive Kernel: %2.5fsecs\n", elapse); // print runtime
        delete[] buffer;
	/*for (int i = 0; i < width_height_channel; i++) {
	  printf("%d ", image[i]);
	  if (i % 99 == 98) printf("\n");
	}
	printf("\n\n");*/
        //std::string gpu_naive_filename_out = "gpu_naive_" + filename_out;   // add prefix "gpu_naive_" for output file name
        stbi_write_jpg(filename_out, width, height, channel, image, 75);   // write out cpu image
    } else if (who == 2) {
        // GPU refactor kernel
        float* buffer = new float[width_height_channel];
        timer.start();
        refactorGPU(image, width, height, channel, sigma_spatial, sigma_range, ROWS_PER_BLOCK, buffer);
        elapse = timer.elapsedTime();
        printf("GPU Refactor Kernel: %2.5fsecs\n", elapse);
        delete[] buffer;
        //std::string gpu_refac_filename_out = "gpu_refac_" + filename_out;
        /*for (int i = 0; i < width_height_channel; i++) {
	  printf("%d ", image[i]);
	  if (i % 99 == 98) printf("\n");
	}
        printf("\n\n");*/
        stbi_write_jpg(filename_out, width, height, channel, image, 75);
    } else {
        printf("Bad choice of Who\n");
    }

    return 0;
}
