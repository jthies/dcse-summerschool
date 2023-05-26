#ifndef MATVEC_HELPERS_H
#define MATVEC_HELPERS_H

 /* declaration for the main operation we want to investigate. We will link
    executables with different implementations of this function to try them out.

    The function computes d_b[0:m-1] = d_A*d_x[0:n-1], where all arrays are
    device pointers. A is row-major and of size m*n, m is the number of rows,
    and n the number of columns.
  */
 void matvecProdGPU(float* d_A, float* d_x, float* d_b, long m, long n, long threadsPerBlock);

 /* dummy function to prevent the compiler from optimizing out multiple repeated matvecs on the same vectors */
 void touch(int m, int n, float* d_A, float* d_x, float* d_b);

/* These are simple routines stored in a separate source file as they are not really
 * important for understanding this example. */ 

 void checkCudaError(const char *errormsg);

 void fillVector(float *a, int m, int offset);
 void showVector(const char *name, float *a, int m);
/* allocate a vector on the CPU */
 float *initVector(int m);
/* allocate a vector on the GPU using CUDA-managed memory */
 float *initUnifiedVector(int m);

/* allocate a matrix on the CPU */
 void fillMatrix(float *a, int n, int m, int offset);
 void showMatrix(const char *name, float *a, int n, int m);
 float *initMatrix(int n, int m);
/* allocate a matrix on the GPU using CUDA-managed memory */
 float *initUnifiedMatrix(int n, int m);

 void freeCudaUnified(float *a);

#endif
