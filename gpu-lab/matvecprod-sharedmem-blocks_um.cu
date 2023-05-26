/* Matrix Vector Multiplication example on Cuda using straightforward stepping over rows/columns
 * 
 * Performs the operation : b = A * x
 * 
 * - 'x' and 'b' are two vectors with size dim;
 * - 'A' is a square matrix with size dim x dim;
 * 
 * Kees Lemmens, Nov 2008, Sept 2010, April 2023
 * Modified by Jonas Thies May 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

#include "cuda.h"

#include "matvec_helpers.h"

void clearVector(float *a, int m)
{  long y;

   for(y=0; y<m; y++)
     a[y] = 0;
   // use memset or calloc later
}

__device__ void ldebug_d(const char *message, long arg1, long arg2, float arg3, float arg4)
{
#if DEBUG > 2
      printf(message, arg1, arg2, arg3, arg4);
#endif
}

__global__ void d_matvecProdCuda(float *d_A, float *d_x, float *d_b, long dim)
{
   long x,y;
   extern __shared__ float l_b[];

   //Each block computes the matrix vector product of a single element of vector 'x'
   // BlockIdx 0 takes vector element 0 etc.

   //ldebug_d("blockIdx %d, threadIdx %d: d_A[blockIdx.x * dim + threadIdx.x]=%lf (dummy=%lf)\n",
   //    blockIdx.x, threadIdx.x,  d_A[blockIdx.x * dim + threadIdx.x], 0);

   if(threadIdx.x == 0)
      d_b[blockIdx.x] = 0;

   for(y = 0; y<dim; y +=blockDim.x) // use a for loop in case vector size dim is larger than blockDim.x
   {
     __syncthreads();
      l_b[threadIdx.x] = d_A[(blockIdx.x * dim) + y + threadIdx.x] * d_x[y + threadIdx.x];
     __syncthreads();

     //ldebug_d("blockIdx %d, threadIdx %d: l_b[]=%lf (dummy=%lf)\n",
     //         blockIdx.x, threadIdx.x, l_b[threadIdx.x], 0);

     // sum the elements of all threads within this block by splitting up until we have a single result
     for(x=blockDim.x/2; x>0; x=x/2)     // This requires blockDim to be a power of 2
     {
        if(threadIdx.x  < x)
        { l_b[threadIdx.x] += l_b[threadIdx.x + x];

          //ldebug_d("reduction: blockIdx %d, threadIdx %d: l_b[tidx]=%lf,l_b[tidx+1/2blk]=%lf\n",
          //         blockIdx.x, threadIdx.x, l_b[threadIdx.x], l_b[threadIdx.x + x]);
        }
        __syncthreads();
     }

     if(threadIdx.x == 0)
     {
        atomicAdd(&d_b[blockIdx.x], l_b[0]);
        //ldebug_d("blockIdx %d, threadIdx %d, l_b[0]=%lf, d_b[blockIdx.x]=%lf\n",
        //         blockIdx.x,threadIdx.x, l_b[0], d_b[blockIdx.x]);
     }
   }
}

void matvecProdGPU(float *d_A, float *d_x, float *d_b, long m, long n, long threadsPerBlock)
{

   assert(m==n);
   long dim=m;
   
   // Use "gridSize" blocks and "blockSize" threads per block:
   // Each block processes one element of the vector, so number of blocks (gridsize) is dim.

   dim3 gridDim(dim, 1);
   dim3 blockDim(threadsPerBlock, 1);

//   fprintf(stderr,"Matrix vector product with dim = %ld, nr of blocks = %u, nr of threads per block = %u\n",
//           dim,gridDim.x, blockDim.x);
   
   d_matvecProdCuda <<<gridDim, blockDim, blockDim.x * sizeof(float)>>> (d_A, d_x, d_b, dim);
   
   cudaGetLastError();
   checkCudaError("Kernel execution error !");
}

