/*
 * Sequential matrix vector product example for the GPU lab:
 *
 * Performs the operation : y = A * x
 *
 * - 'A' is a square matrix with size NxN;
 * - 'x' and 'y' are vectors of size N;
 *
 * April 2023; Kees Lemmens; TWA-EWI TU Delft.
 * Adapted May 2023: Jonas Thies, TU Delft
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef USE_MKL
#include <mkl_cblas.h>
#elif defined(USE_BLAS)
#include <cblas.h>
#endif

#include "matvec_helpers.h"

//typedef double real;
typedef float real;

/* These are simple routines stored in a separate source file as they are not really
 * important for understanding this example. */ 

#ifdef USE_OPENMP
/* TODO: insert OpenMP pragma's for parallelization on the CPU */
void matvecprod(real *A, real const* x, real *b, int m, int n)
{
   int i,j;

   for(i=0;i<m;i++)
   {
      b[i] = 0;
      for(j=0;j<n;j++)
      {
         b[i] += x[j] * A[i*n+j];
      }
   }
}
#elif defined(USE_BLAS) || defined(USE_MKL)
/* This version calls the BLAS routine SGEMV on the CPU */
void matvecprod(float *A, float const* x, float *b, int m, int n)
{
   float alpha = 1.0f, beta = 0.0f;
   int incx = 1, incb = 1;
   int lda=n; // leading dimensino of A (note that A is stored in row-major order!)
   cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, lda, x, incx, beta, b, incb);
}
#elif defined(USE_OMP_TARGET)
/* TODO: insert OpenMP pragma's for offloading to the GPU
         Statements you may want to try are:
                pragma omp target (with the map clause to indicate required data movement)
                pragma omp teams distribute for parallel execution among multiple GPU processors
                pragma omp parallel for for parallelization among the threads of each processor
 */
void matvecprod(real *A, real const* x, real *b, int m, int n)
{
   for(int i=0;i<m;i++)
   {
      b[i] = 0;
      for(int j=0;j<n;j++)
      {
         b[i] += x[j] * A[i*n+j];
      }
   }
}
#endif

int main(int argc, char *argv[])
{
   real *A, *x, *b;
   struct timeval ti1,ti2;
   long dim = 4;
   int num_runs=30;

   real runtime, bandwidth;

   char label[3];

   sprintf(label,"CPU");

   if(argc >=2 )sscanf(argv[1],"%ld",&dim);
   if(argc >=3 )sscanf(argv[2],"%d",&num_runs);
   
   fprintf(stderr,"Matrix Vector product with dim = %ld, num_runs=%d\n",dim, num_runs);
   
   A = initMatrix(dim,dim);
   x = initVector(dim);
   b = initVector(dim);
   
   fillMatrix(A,dim,dim, 10);
   fillVector(x,dim, 1);

{
   gettimeofday(&ti1,NULL); /* read starttime in t1 */

#ifdef USE_OMP_TARGET
  sprintf(label,"GPU");
#endif
   for (int i=0; i<num_runs; i++)
   {
     matvecprod(A,x,b,dim,dim);
   }
   gettimeofday(&ti2,NULL); /* read endtime in t2 */

   showMatrix("A",A,dim,dim);
   showVector("x",x,dim);
   showVector("b",b,dim);

   runtime = (ti2.tv_sec - ti1.tv_sec ) + 1e-6*(ti2.tv_usec - ti1.tv_usec);
   // memory traffic: load matrix (n*n) and vector (n), load+store result
   // (unless a streaming ("non-temporal") store is used)
   bandwidth = ((double)(dim*dim+3*dim)*sizeof(real))/runtime*1e-9*num_runs;

   fflush(stderr);
   fprintf(stderr,"\n%s: average run time = %f secs.\n",label, runtime/num_runs);
   fprintf(stderr,"%s: memory bandwidth = %f GB/s.\n",label, bandwidth);
}
   return 0;
}
