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

//typedef double real;
typedef float real;

/* These are simple routines stored in a separate source file as they are not really
 * important for understanding this example. */ 

extern real *initVector(int m);
extern void freeVector(real *ptr);
extern void fillVector(real *a, int m, int offset);
extern void showVector(char *name, real *a, int m);

extern real **initMatrix(int n, int m);
extern void freeMatrix(real **mtr);
extern void fillMatrix(real **a, int n, int m, int offset);
extern void showMatrix(char *name, real **a, int n, int m);

#if defined(USE_BLAS) || defined(USE_MKL)
void matvecprod(float **A, float const* x, float *b, int m, int n)
{
   float alpha = 1.0f, beta = 0.0f;
   int incx = 1, incb = 1;
   int lda=n; // leading dimensino of A (note that A is stored in row-major order!)
   cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, &A[0][0], lda, x, incx, beta, b, incb);
}
#else
void matvecprod(real **A, real const* x, real *b, int m, int n)
{
   int i,j;

#pragma omp parallel for schedule(static)
   for(i=0;i<m;i++)
   {
      b[i] = 0;
#pragma omp simd
      for(j=0;j<n;j++)
      {
         b[i] += x[j] * A[i][j];
      }
   }
}
#endif

int main(int argc, char *argv[])
{
   real **A, *x, *b;
   struct timeval ti1,ti2;
   long dim = 4;
   // read matrix size from command line if provided, else use default value of 4.
   if (argc>1) dim=atol(argv[1]);
   int num_runs=30;

   real runtime, bandwidth;

   if(argc >=2 )
     sscanf(argv[1],"%ld",&dim);
   
   fprintf(stderr,"Matrix Vector product with dim = %ld\n",dim);
   
   A = initMatrix(dim,dim);
   x = initVector(dim);
   b = initVector(dim);
   
   fillMatrix(A,dim,dim, 10);
   fillVector(x,dim, 1);

   gettimeofday(&ti1,NULL); /* read starttime in t1 */
   
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
   fprintf(stderr,"\nCPU : average run time = %f secs.",runtime);
   fprintf(stderr,"\nCPU : memory bandwidth = %f GB/s.\n",bandwidth);
   
   return 0;
}
