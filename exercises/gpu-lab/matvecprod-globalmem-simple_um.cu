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

#include "cuda.h"
#include "matvec_helpers.h"

__global__ void d_matvecProdCuda(float *d_A, float *d_x, float *d_b, long dim)
{
   float bs=0;
   long i,j;
   
   // Each thread computes the matvec product for only one single vector element
 
   i = (blockIdx.x * blockDim.x) + threadIdx.x; // row index for this thread (=index of col vector)!
   //printf("i = %ld, threadIdx.x = %ld, blockIdx.x = %ld\n",i,threadIdx.x,blockIdx.x);

   if(i<dim)
   {
      for(j=0; j < dim; j++) // compute product for row element in A x column element in B
      {
         bs += d_A[i * dim + j] * d_x[j];
         // printf("matrix index = %ld\n", i * dim + j);
      }
      
      // Store result back into global memory for C:
      d_b[i] = bs;
   }
}

void matvecProdGPU(float *d_A, float *d_x, float *d_b, long m, long n, long threadsPerBlock)
{
   assert(m==n);
   long dim=m;
   // Use "gridDim" blocks and "blockDim" threads per block:
   dim3 gridDim, blockDim;

   gridDim  = {(uint)(dim%threadsPerBlock ? dim/threadsPerBlock + 1: dim/threadsPerBlock), 1};
   blockDim = {(uint)threadsPerBlock,                  1};
   
//   fprintf(stderr,"Matrix vector product with dim = %ld, nr of blocks = %u, nr of threads per block = %u\n",
//           dim,gridDim.x, blockDim.x);

   d_matvecProdCuda <<<gridDim, blockDim>>> (d_A, d_x, d_b, dim);
 
   cudaGetLastError();
   checkCudaError("Kernel execution error !");
}

