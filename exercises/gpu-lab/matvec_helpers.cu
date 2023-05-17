/* Some common routines for allocating matrices,
 * filling them with some data and printing them.
 * 
 * Kees Lemmens, June 2009
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>

void checkCudaError(const char *errormsg)
{
   cudaError_t error = cudaGetLastError();
   if (error != cudaSuccess)
   {
      fprintf (stderr, "%s\n",errormsg);
      fprintf (stderr, "Cuda: %s\n",cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

// Vector routines:

void fillVector(float *a, int m, int offset)
{  long y;
   
   for(y=0; y<m; y++)
     a[y] = (float) (y + offset);
}

void showVector(const char *name, float *a, int m)
{ 
   long y;

# if (DEBUG > 0)
   for(y=0; y<m; y++)
# else
   y = m - 1;
# endif
   {
      printf("%s[%02ld]=%6.2f  ",name,y,a[y]);
   }
   printf("\n");
}
   
float *initHostVector(int m)
{
   float *ptr = 0;

   cudaMallocHost(&ptr, m * sizeof(float)); // rows
   checkCudaError("Malloc for vector on host failed !");
   if(ptr == NULL)
   {
      fprintf(stderr,"Malloc for Host vector resulted in NULL pointer!\n");
      exit(EXIT_FAILURE);
   }
   
   return ptr;
}

float *initCudaVector(int m)
{
   float *ptr = 0;
   
   cudaMalloc(&ptr, m * sizeof(float));
   checkCudaError("Malloc for vector on device failed !");
   if(ptr == NULL)
   {
      fprintf(stderr,"Malloc for Cuda vector resulted in NULL pointer!\n");
      exit(EXIT_FAILURE);
   }

   return ptr;
}

void copytoCudaVector(float *d_a, float *h_a, int m)
{
   cudaMemcpy(d_a, h_a, m * sizeof(float), cudaMemcpyHostToDevice);
   checkCudaError(" Vector copy to device failed !");
}

void copyfromCudaVector(float *h_a, float *d_a, int m)
{
   cudaMemcpy(h_a, d_a, m * sizeof(float), cudaMemcpyDeviceToHost);
   checkCudaError("Vector copy from device failed !");
}

// Matrix routines:

void fillMatrix(float *a, int n, int m, int offset)
{  long x,y;
   
   for(y=0; y<m; y++)
     for(x=0; x<n; x++)
       a[y*n + x] = (float) (x + y + offset);
}

void showMatrix(const char *name, float *a, int n, int m)
{ 
   long x,y;

# if (DEBUG > 0)
   for(y=0; y<m; y++)
# else
   y = m - 1;
# endif
   {
# if (DEBUG > 1)
      for(x=0; x<n; x++)
# else
      // print only the last element
      x = n - 1;
# endif
      {
        printf("%s[%02ld][%02ld]=%6.2f  ",name,y,x,a[y*n + x]);
      }
      printf("\n");
   }
}
   
float *initHostMatrix(int n, int m)
{
   float *ptr = 0;

   cudaMallocHost(&ptr, n * m * sizeof(float)); // rows x columns
   checkCudaError("Malloc for matrix on host failed !");
   if(ptr == NULL)
   {
      fprintf(stderr,"Malloc for Host matrix resulted in NULL pointer!\n");
      exit(EXIT_FAILURE);
   }

   return ptr;
}

float *initCudaMatrix(int n, int m)
{
   float *ptr = 0;
   
   cudaMalloc(&ptr, n * m * sizeof(float));
   checkCudaError("Malloc for matrix on device failed !");
   if(ptr == NULL)
   {
      fprintf(stderr,"Malloc for Cuda matrix resulted in NULL pointer!\n");
      exit(EXIT_FAILURE);
   }

   return ptr;
}

void copytoCudaMatrix(float *d_a, float *h_a, int n, int m)
{
   cudaMemcpy(d_a, h_a, n * m * sizeof(float), cudaMemcpyHostToDevice);
   checkCudaError(" Matrix copy to device failed !");
}

void copyfromCudaMatrix(float *h_a, float *d_a, int n, int m)
{
   cudaMemcpy(h_a, d_a, n * m * sizeof(float), cudaMemcpyDeviceToHost);
   checkCudaError("Matrix copy from device failed !");
}

// Routines for both:

void freeCudaHost(float *a)
{
   cudaFreeHost(a);
   checkCudaError("Memory free error on host !");
}

void freeCudaDevice(float *a)
{
   cudaFree(a);
   checkCudaError("Memory free error on device !");
}

// Dummy routine to prevent optimizing out all matvecs
void touch(int n, int m, float* A, float* d_x, float* d_b)
{
  return;
}
