#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
/* ---------
the goal is to write a kernel function for the second refactored loop
this loop has been refactored to be parallelizable: each column of computation
can be paralleled. Within each column, computation is sequential as later row depends
on previous row. 
The first method is to treat each column of processing as a thread. As column data is not
got from a consecutive block of memory therefore the cost for each thread to get its
data from global memory might be high.
The second way is: We might transpose the matrix and then speed up by GPU. But as transpoing a 
matrix has an overhead of O(n*n), it might not be a good strategy. 
So in principle, GPU acceleration of this part might not yield efficient results. But here we 
still stick to the first method and try to see if GPU can speed up the computation.  --------*/

__global__ void secondKernel (
  
    unsigned char * img, float * img_temp, float * img_out_f, float * in_factor, float * map_factor_b,
    float* range_table, int width, int height, int channel, int width_channel,float alpha, float inv_alpha_
){
    /*----
    there are a number of width tasks to parallize. (width is the number of pixels per row)
    the width might be large like 5760 or even larger, as there might be a limit of the
    number of threads per block. We might propose a (width/1024) 1D blocks(suppose 1024 threads/b)
    ---- */
        int index =  blockIdx.x * blockDim.x + threadIdx.x;
    //initialize parameters
        float * ycy, * ypy, * xcy, * ycf, * ypf, * xcf;
        unsigned char * tpy, *tcy;
        if(index >= width) return;
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
                // *ycf_++ = inv_alpha_*(*xcf_++) + alpha_*(*ypf_++); 
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
