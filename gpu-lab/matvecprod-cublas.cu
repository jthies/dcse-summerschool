/* Matrix Vector Multiplication example using CuBLAS
 * 
 * Performs the operation : b = A * x
 * 
 * - 'x' and 'b' are two vectors with size dim;
 * - 'A' is a square matrix with size dim x dim;
 *
 * Jonas Thies May 2023
 * Adaped from Kees Lemmens examples, Nov 2008, Sept 2010, April 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda.h>
#include <cublas.h>

#include "matvec_helpers.h"

void matvecProdGPU(float *d_A, float *d_x, float *d_b, long m, long n, long threadsPerBlock)
{
   float alpha=1.0, beta=0.0;
   /* note: because we use row-major storage of A, we have to ask for the transposed operation
      from CuBLAS, which also means we need to swap the row and column dimensions...
    */
   cublasSgemv('T', n, m, alpha, d_A, n, d_x, 1, beta, d_b, 1);
   cudaGetLastError();
   checkCudaError("Kernel execution error !");
}

