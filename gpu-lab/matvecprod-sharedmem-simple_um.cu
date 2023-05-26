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

/* These are simple routines stored in a separate source file as they are not really
 * important for understanding this example. */ 
#include "matvec_helpers.h"

__global__ void d_matvecProdCuda(float *d_A, float *d_x, float *d_b, long dim)
{
   float bs=0;
   long i, j, k, r;
   long dim2 = dim * dim;
   extern __shared__ float l_x[];
   
   // Each thread computes the matvec product for only one single vector element

   i = (blockIdx.x * blockDim.x) + threadIdx.x; // row index this thread (=index of col vector)!

   for(j=0; j < dim; j+=blockDim.x) // compute product for row el in A x col el in B
   {      
      __syncthreads();  // MUST sync here as well if the nr of threads > 32 (warp size!!)
      if(j + threadIdx.x < dim)
        l_x[threadIdx.x] = d_x[j + threadIdx.x]; // Copy vector element to shared memory
      __syncthreads();

      r = i * dim + j;      
      for(k=0; k<blockDim.x; k++)
      {
         // column index = i
         // row index = j + k
         if(r + k < dim2) // outside matrix A
         {
            bs += d_A[r + k] * l_x[k];        // from shared memory
            // bs += d_A[r + k] * d_x[j + k]; // from global memory
            // if( d_x[j + k] != l_x[k] )
            //   printf("Alarm: matrix index=%ld (i=%ld, j=%ld, k=%ld), A=%f, d_x=%f, l_x=%f\n",
            //     r + k, i, j, k, d_A[r + k], d_x[j + k], l_x[k]);
         }
      }
   }
   
   // Store result back into global memory:
   if(i<dim)
     d_b[i] = bs;
}


void matvecProdGPU(float *d_A, float *d_x, float *d_b, long m, long n, long threadsPerBlock)
{
   assert(m==n);
   long dim=m;
   // Use "gridDim" blocks and "blockDim" threads per block:
   dim3 gridDim(dim%threadsPerBlock ? dim/threadsPerBlock + 1: dim/threadsPerBlock, 1);
   dim3 blockDim((uint)threadsPerBlock,                    1);

//   fprintf(stderr,"Matrix vector product with dim = %ld, nr of blocks = %u, nr of threads per block = %u\n",
//           dim, gridDim.x, blockDim.x);

   d_matvecProdCuda <<<gridDim, blockDim, blockDim.x * sizeof(float)>>> (d_A, d_x, d_b, dim);

   cudaGetLastError();
   checkCudaError("Kernel execution error !");
}

