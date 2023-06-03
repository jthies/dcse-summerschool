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

/* multiply C = A*B, where A, B and C are stored in row-major order,
   A is (n x n), B and C are n x k (n rows and k columns).
   Note that due to the fact that BLAS routines assume column-major ordering,
   we need to pretend we do the operation C = (A^T*B^T), where OP(B)=B^T is n x k.
*/
void matmatProdGPU(float const* d_A, float const* d_B, float *d_C, long n, long k, long threadsPerBlock)
{

   float alpha=1.0, beta=0.0;
   /* note: because we use row-major storage of A, we have to ask for the transposed operation
      from CuBLAS, which also means we need to swap the row and column dimensions...
    */
   cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_T, n, n, k, alpha, d_A, n, d_B, k, beta, d_C, k);
   cudaGetLastError();
   checkCudaError("Kernel execution error !");
}

