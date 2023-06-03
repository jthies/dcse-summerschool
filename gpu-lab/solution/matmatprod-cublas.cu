/* Matrix Vector Multiplication example using CuBLAS_T

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
 */
void matmatProdGPU(float const* d_A, float const* d_B, float *d_C, long n, long k, long threadsPerBlock)
{

   float alpha=1.0, beta=0.0;

   /* note: because we use row-major storage of A, B and C,we have to be careful here:
   CUBLAS assumes the are all in column-major storage, so it sees the matrices as transposed arrays
   (if passed the correct dimensions and strides). From the input matrices A^T and B^T, we can construct
   C^T as C^T = (A*B)^T = B^T*A^T, so we just swap the matrices and we're done.
   The strides (or leading dimensions) remain the number of columns in the original matrices A, B, C.

    */
   cublasSgemm('N', 'N', k, n, n, alpha, d_B, k, d_A, n, beta, d_C, k);
   cudaGetLastError();
   checkCudaError("Kernel execution error !");
}

