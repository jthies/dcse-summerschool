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

int main(int argc, char** argv)
{
   struct timespec ti1,ti2;
   double runtime, bandwidth;
   long dim = 4;
   int num_runs=300;
   float *d_A = 0, *d_x = 0, *d_b = 0;

   if(argc >=2 ) sscanf(argv[1],"%ld",&dim);

   if(argc >=3 ) sscanf(argv[2],"%d",&num_runs);

   clock_gettime(CLOCK_REALTIME,&ti1);

   /* Allocate unified memory for the matrices */
   d_A = initUnifiedMatrix(dim,dim);
   d_x = initUnifiedVector(dim);
   d_b = initUnifiedVector(dim);

   fillMatrix(d_A,dim,dim, 10);
   fillVector(d_x,dim, 1);

   /* Clear last error */
   cudaGetLastError();
   
   /* Performs operation using Cuda kernel above */
   for (int i=0; i<num_runs; i++)
   {
     matvecProdGPU(d_A, d_x, d_b, dim, dim, MAXBLOCKSIZE);
     touch(dim, dim, d_A, d_x, d_b);
   }
   /* Read the result back */
   cudaDeviceSynchronize();

   clock_gettime(CLOCK_REALTIME,&ti2);
   runtime = (ti2.tv_sec - ti1.tv_sec) + 1e-9*(ti2.tv_nsec - ti1.tv_nsec);
   // memory traffic: load matrix (n*n) and vector (n), load+store result
   // (unless a streaming ("non-temporal") store is used)
   bandwidth = ((double)(dim*dim+3*dim)*sizeof(float))/runtime*1e-9*num_runs;

   showMatrix("A",d_A,dim,dim);
   showVector("x",d_x,dim);
   showVector("b",d_b,dim);

   /* Memory clean up */

   freeCudaUnified(d_A);
   freeCudaUnified(d_x);
   freeCudaUnified(d_b);

   fflush(stderr);
   fprintf(stderr,"\nCuda: average run time = %f secs.\n",runtime/num_runs);
   fprintf(stderr,"Cuda: memory bandwidth = %f GB/s.\n",bandwidth);
   
   return EXIT_SUCCESS;
}
