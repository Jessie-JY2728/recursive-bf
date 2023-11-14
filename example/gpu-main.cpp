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
    if (argc != 3)
	{
		printf("Usage:\n");
		printf("--------------------------------------------------------------------\n\n");
		printf("rbf filename_in rows_per_block \n");
        printf("Where rows_per_block is how many rows a block will process\n");
		printf("--------------------------------------------------------------------\n");
		return(-1);
	}
    //const char *filename_out = argv[1];
	const char *filename_in = argv[1];

    std::string filename_out(filename_in);
    int ROWS_PER_BLOCK = atoi(argv[2]);
	float sigma_spatial = 0.03;
	float sigma_range = 0.1;
    int width, height, channel;
    unsigned char *image = stbi_load(filename_in, &width, &height, &channel, 0);
    if (!image) {
        printf("Low Rating stb has FAILED to load Input Image. SAD.");
        exit(1);
    }
    printf("Loaded image: w=%d, h=%d, c=%d", width, height, channel);
    // char* image: input file

    int width_height = width * height;
    int width_height_channel = width_height * channel;
    int width_channel = width * channel;

    Timer timer;
    float elapse;

    // CPU version
    unsigned char * image_out = 0;
    float *buffer = new float[(width_height_channel + width_height + width_channel + width) * 2];
    timer.start();  // start timer
	recursive_bf(image, image_out, sigma_spatial, sigma_range, width, height, channel, buffer); // use original rbf
    elapse = timer.elapsedTime(); // runtime
	printf("External Buffer: %2.5fsecs\n", elapse); // print runtime
	delete[] buffer;    // clean up
    std::string cpu_filename_out = "cpu_" + filename_out;   // add prefix "cpu_" for output file name
    stbi_write_jpg(cpu_filename_out.c_str(), width, height, channel, image_out, 100);   // write out cpu image


    // GPU naive kernel
    timer.start();
    invokeNaiveKernel(image, width, height, channel, sigma_spatial, sigma_range, ROWS_PER_BLOCK);
    elapse = timer.elapsedTime();   // runtime



    return 0;
}