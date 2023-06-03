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
#include <string.h>
#include <time.h>

#include "cuda.h"

#include "matvec_helpers.h"

// Nr of threads per threadblock or blocksize (MAX = 16x16 for NVS290)
#define MAXBLOCKSIZE 256

/* multiply C = A*B, where A, B and C are stored in row-major order,
   A is (n x n), B and C are n x k (n rows and k columns).
   Note that due to the fact that BLAS routines assume column-major ordering,
   we need to pretend we do the operation C = (A^T*B^T), where OP(B)=B^T is n x k.
*/
void matmatProdGPU(float const* d_A, float const* d_B, float *d_C, long n, long k, long threadsPerBlock);

int main(int argc, char** argv)
{
   struct timespec ti1,ti2;
   double runtime, bandwidth, floprate, intensity;
   long dim_n = 4, dim_k=1;
   int num_runs=300;
   float *d_A = 0, *d_B = 0, *d_C = 0;

   if(argc >=2 ) sscanf(argv[1],"%ld",&dim_n);

   if(argc >=3 ) sscanf(argv[2],"%ld",&dim_k);

   if(argc >=4 ) sscanf(argv[3],"%d",&num_runs);

   intensity = (double)(dim_n*dim_n*dim_k*2.0) / ((double)(dim_n*dim_n + 3.0*dim_n*dim_k)*8.0);

   clock_gettime(CLOCK_REALTIME,&ti1);

   /* Allocate unified memory for the matrices */
   d_A = initUnifiedMatrix(dim_n,dim_n);
   d_B = initUnifiedMatrix(dim_n,dim_k);
   d_C = initUnifiedMatrix(dim_n,dim_k);

   fillMatrix(d_A,dim_n,dim_n, 10);
   fillMatrix(d_B,dim_n,dim_k, 10);

   /* Clear last error */
   cudaGetLastError();

   /* Performs operation using Cuda kernel above */
   for (int i=0; i<num_runs; i++)
   {
     matmatProdGPU(d_A, d_B, d_C, dim_n, dim_k, MAXBLOCKSIZE);
     touch(dim_n, dim_k, d_A, d_B, d_C);
   }
   /* Read the result back */
   cudaDeviceSynchronize();

   clock_gettime(CLOCK_REALTIME,&ti2);
   runtime = (ti2.tv_sec - ti1.tv_sec) + 1e-9*(ti2.tv_nsec - ti1.tv_nsec);
   // memory traffic: load matrix (n*n) and vector (n), load+store result
   // (unless a streaming ("non-temporal") store is used)
   bandwidth = ((double)(dim_n*dim_n+3*dim_n*dim_k)*sizeof(float))/runtime*1e-9*num_runs;
   floprate = ((double)(dim_n*dim_n*dim_k*2))/runtime*1e-9*num_runs;

   showMatrix("A",d_A,dim_n,dim_n);
   showMatrix("B",d_B,dim_n,dim_k);
   showMatrix("C",d_C,dim_n,dim_k);

   /* Memory clean up */

   freeCudaUnified(d_A);
   freeCudaUnified(d_B);
   freeCudaUnified(d_C);

   fflush(stderr);
   fprintf(stderr,"\nCuBLAS GEMM (%d x %d x %d): computational intensity = %f Flops/Byte.",dim_n,dim_n,dim_k,intensity);
   fprintf(stderr,"\nCuBLAS GEMM (%d x %d x %d): average run time = %f secs.",dim_n,dim_n,dim_k,runtime/num_runs);
   fprintf(stderr,"\nCuBLAS GEMM (%d x %d x %d): memory bandwidth = %f GB/s.",dim_n,dim_n,dim_k,bandwidth);
   fprintf(stderr,"\nCuBLAS GEMM (%d x %d x %d): performance = %f Gflop/s.\n",dim_n,dim_n,dim_k,floprate);

   return EXIT_SUCCESS;
}
