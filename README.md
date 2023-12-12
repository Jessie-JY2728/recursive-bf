This is a GPU CUDA accelerated version of the Recursive Bilateral Filter program. The original algorithm (developed by Qingxiong Yang) is already pretty fast compared with most edge-preserving filtering methods. The GPU version is ~10x faster than its sequential basis on larger images. 
- takes 1.64 secs to process a 15870 x 7933 RGB image, an 8.31x speedup
- takes 0.79 secs to process a 17707 x 4894 RGB image, a 11.86x speedup

The /include directory contains original and refactored CPU implementation. The /example directory contains the GPU version, and /images contain sample images used for testing. To compile and run the program, do the following commands:

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


## Results
<table>
<tr>
<td><img src="https://cloud.githubusercontent.com/assets/2270240/26041579/7d7c034e-3960-11e7-9549-912685043e39.jpg" width="300px"><br/><p align="center">Original Image</p></td>
<td><img src="https://cloud.githubusercontent.com/assets/2270240/26041586/8b4afb42-3960-11e7-9bd8-62bbb924f1e9.jpg" width="300px"><br/><p align="center">OpenCV's BF (896ms)</p></td>
<td><img src="https://cloud.githubusercontent.com/assets/2270240/26041590/8d08c16c-3960-11e7-8a0c-95a77d6d9085.jpg" width="300px"><br/><p align="center">RecursiveBF (18ms)</p></td>
</tr>
<tr>
<td></td>
<td><img src="https://cloud.githubusercontent.com/assets/2270240/26041583/86ea7b22-3960-11e7-8ded-5109b76966ca.jpg" width="300px"><br/><p align="center">Gaussian Blur</p></td>
<td><img src="https://cloud.githubusercontent.com/assets/2270240/26041584/88dfc9b4-3960-11e7-8c9d-2634eac098d0.jpg" width="300px"><br/><p align="center">Median Blur</p></td>
</tr></table>
