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

   /* note: because we use row-major storage of A, B and C, we have to ask for the transposed operation
      from CuBLAS, which also means we need to swap the row and column dimensions.
      C = A*B = (B^T*A^T)^T

      According to the documentation, we need to pass in AA, BB and CC such that CC = opA(AA)*opB(BB), and N, M, K s.t. opA(AA) is MxK, op(BB) is KxN and CC is MxN
      For us this means: AA^T=B^T is k x n, BB^T=A^T is n x n, and CC=C^T is k x n
    */
   cublasSgemm('T', 'T', n, k, n, alpha, d_B, n, d_A, n, beta, d_C, k);
   cudaGetLastError();
   checkCudaError("Kernel execution error !");
}

