/* Some common routines for allocating matrices in unified memory
 * 
 * Kees Lemmens, May 2023
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include "matvec_helpers.h"

// Vector routines:

float *initUnifiedVector(int m)
{
   float *ptr = 0;

   cudaMallocManaged(&ptr, m * sizeof(float)); // rows
   checkCudaError("Malloc for unified vector failed !");
   if(ptr == NULL)
   {
      fprintf(stderr,"Malloc for Unified vector resulted in NULL pointer!\n");
      exit(EXIT_FAILURE);
   }

   return ptr;
}

// Matrix routines:

float *initUnifiedMatrix(int n, int m)
{
   float *ptr = 0;

   cudaMallocManaged(&ptr, n * m * sizeof(float)); // rows x columns
   checkCudaError("Malloc for unified matrix failed !");
   if(ptr == NULL)
   {
      fprintf(stderr,"Malloc for Unified matrix resulted in NULL pointer!\n");
      exit(EXIT_FAILURE);
   }

   return ptr;
}

// Routines for both:

void freeCudaUnified(float *a)
{
   cudaFree(a);
   checkCudaError("Memory free error for unified memory !");
}
