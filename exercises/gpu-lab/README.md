# GPU lab: BLAS 1/2/3 Operations in an Iterative Eigenvalue Solver

In this exercise, we will compute the smallest eigenvlaue of the Laplace operator discretized using
a pseudo-spectral method. The operator is represented by a dense matrix $L$ in one space dimension,
and by $I \mtimes D + D \mtimes I$ in 2D.

- Problem 1: Lanczos method for computing the smallest eigenvalue of a dense matrix
    1. implement DGEMV using CUDA and a BLAS3-call, compare performance
    2. implement dot and axpby operations using CUDA and/or a BLAS1 call, compare performance
    3. put it all together in a given Lanczos code skeleton
    4. profile the complete eigensolver using nvprof
- Problem 2: Same for 2D version that can be written as (I (x) C) + (C (x) I)
    1. Extend your own DGEMV to DGEMM, compare to BLAS3-call
    2. Implement 2D operator using DGEMM
    3. compute smallest eigenvalue of the 2D operator, use nvprof to see what are the bottlenecks and hotspots now

# Your tasks

(detailed description of the steps they should take, with references to code/scripts/commands)

- what is the computational intensity of the operation (in Flops/Byte)?
- implement GEMV using CUDA, compare performance with CUBLAS and performance model
- implement axpy and dot, run simple tests and benchmarks
- take a Lanczos code, do memory allocation using CUDA managed memory, compute largest eigenvalue of a dense matrix
- compare overall performance CPU/GPu
- Extend your operator to a 2D problem of the form (I (x) C) + (C (x) I) and implement it using GEMM, compare again CPU/GPU.
