/*
 * Sequential matrix vector product example for the GPU lab:
 * 
 * Performs the operation : y = A * x
 * 
 * - 'A' is a square matrix with size NxN;
 * - 'x' and 'y' are vectors of dsize N;
 *
 * April 2023; Kees Lemmens; TWA-EWI TU Delft.
*/  

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

//typedef double real;
typedef float real;

real *initVector(int m)
{
   real *ptr;
   
   ptr = (real  *) calloc(m, sizeof(real)); // rows x columns
   
   if(ptr == NULL)
   {
      fprintf(stderr,"Malloc for vector failed !\n");
      exit(1);
   }
   
   return ptr;
}

/* Next function frees a matrix allocated with initMatrix() */
void freeVector(real *ptr)
{
   free(ptr);
}

void fillVector(real *a, int m, int offset)
{  long y;
   
#pragma omp parallel for schedule(static)
   for(y=0; y<m; y++)
       a[y] = (real) (y + offset);
}

/* Next function simply prints a vector on the screen. */
void showVector(char const* name, real *a, int m)
{
   long y;

# if (DEBUG > 0)
   for(y=0; y<m; y++)
# else
   y = m - 1;
# endif
   {
      printf("%s[%02lu]=%6.2f ",name,y,a[y]);
   }
   printf("\n");
}

real *initMatrix(int n, int m)
{
   int t;
   real *ptr;

   ptr = (real  *) calloc(n * m, sizeof(real)); // rows x columns

   if(ptr == NULL)
   {
      fprintf(stderr,"Malloc for matrix strip failed !\n");
      exit(1);
   }
   return ptr;
}

/* Next function frees a matrix allocated with initMatrix() */
void freeMatrix(real *mtr)
{
   free(mtr);
}

void fillMatrix(real *a, int n, int m, int offset)
{  long x,y;
   
#pragma omp parallel for schedule(static)
   for(y=0; y<m; y++)
     for(x=0; x<n; x++)
       a[y*n+x] = (real) (x + y + offset);
}

/* Next function simply prints a matrix on the screen. */
void showMatrix(char const* name, real *a, int n, int m)
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
         printf("%s[%02lu][%02lu]=%6.2f  ",name,y,x,a[y*n+x]);
      }
      printf("\n");
   }
}

