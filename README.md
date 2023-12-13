This is a GPU CUDA accelerated version of the Recursive Bilateral Filter program. The original algorithm (developed by Qingxiong Yang) is already pretty fast compared with most edge-preserving filtering methods. The GPU version is ~10x faster than its sequential basis on larger images. 
- takes 1.64 secs to process a 15870 x 7933 RGB image, an 8.31x speedup
- takes 0.79 secs to process a 17707 x 4894 RGB image, a 11.86x speedup

The `/include` directory contains original and refactored CPU implementation. The `/example` directory contains the GPU version, and `/images` contain sample images used for testing. To compile and run the program, do the following commands:

```
cd example
nvcc -o <executable> gpu-main.cpp gpu-kernels.cu
./<executable> <filename_in> <filename_out> <rows_per_block> <who>
```
where <rows_per_block> is the block dimension, and 
- who=0: CPU version
- who=1: GPU naive version
- who=2: GPU refactored version

for example:
```
cd example
nvcc -o rbf gpu-main.cpp gpu-kernels.cu
./rbf images/1.jpeg 1_gpu.jpg 32 2
```
