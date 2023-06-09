# GPU lab: Dense Matrix-Vector Multiplication on GPUs

In this exercise, we will investigate the performance of dense  general matrix-vector (GEMV) product
with various GPU implementations:
```math
b=A\cdot x, A\in\mathbb{R}^{m \times n}, x \in \mathbb{R}^{m}, b \in \mathbb{R}^{n}.
```

- Note that you should compile the GPU examples below on the GPU node itself, therefore,
the ``module`` and ``make`` commands are included in the corresponding job script.
- Before compiling the CPU version, type ``source env.sh`` to load the right compilers and libraries.
- Test that your implementation produces correct results by comparing the printed values of b with the ones
obtained with the sequential and/or BLAS implementations.
- If you get stuck, you may look in the ``solution`` folder for oworking versions.

## What you will learn/practice here

- offloading computations to the GPU with OpenMP,
- keeping data on the GPU between calls,
- calling optimized GPU library routines (CuBLAS here)
- estimating the performance difference between CPU and GPU

## Your tasks

1. What is the computational intensity of the operation (in Flops/Byte)? Is this operation compute- or memory-bound?
2. Insert OpenMP pragma's to parallelize the first version of ``matvecprod`` in ``matvecprod-cpu.c`` (in the ``USE_OPENMP`` block).
   Compile the code using ``make cpu`` (along with a version that uses OpenBLAS and one that uses Intel MKL).
   Benchmark all three versions using ``sbatch run-cpu.sh``. What do you observe?
3. The following three sections can be worked on independently, decide what you are most interested in.
Time during the lab may be too short to finish everything.

## Offloading operations to the GPU with OpenMP

- Insert OpenMP pragma's for GPU offloading in the second version of ``matvecprod`` (in the ``USE_OMP_TARGET`` section).
   The main construct you should use is ``#pragma omp target``, with the ``map`` clause to define which data should be copied
   to and from the device. For example, you could compute the (squared) norm of a vector ``a`` like this:
   ```c++
   norm=0;
   #pragma omp target map(to: n, a[0:n]) map(tofrom: norm)
   #pragma omp teams distribute parallel for reduction(+:norm)
   for (int i=0; i<n; i++) norm+=a[i]*a[i];
   ```
   Use the script ``compile-and-run-gpu.sh`` to test and benchmark the implementation.  
- Improve the implementation by moving the data transfers outside the benchmark loop.
This can be achieved using a ``data`` statement, e.g.:
```c++
#pragma omp target data map(to:a[n]) map(from: norm)
for (int i=0; i<num_runs; i++)
{
// omp target code using a on the device
}
```

## GEMV performance and different CUDA implementations

- We have provided several CUDA-based implementations. The timing and bandwidth results that the driver reports
   includes transferring matrix and vectors to the device, and the result vector back. Can you determine the time
   for these data transfers using one of the drivers?
- Choose suitable values for <dim> and <num_runs>, and compare the memory bandwidth achieved with your OpenMP variant,
   the various CUDA implementations and the one using cuBLAS.

## Extension to Matrix-Matrix (GEMM) with CUBLAS

- Extend the cuBLAS driver to perform a general matrix-matrix (GEMM) operation instead of GEMV:
   ```math
   C = A\cdot B, A\in \mathbb{R}^{m\times n}, B\in\mathbb{R}^{n\times k}, C\in\mathbb{R}^{m\times k}.
   ```
   and to print both memory bandwidth and performance (in GFlop/s). Increase ``k`` starting from 1 in a series
   of runs and observe how the computational intensity changes.
   The cublas 'gemm' documentation can be found [here](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm).  
   **Caveat:** Since CuBLAS assumes column-major storage, you are actually passing in the transposed of A, and B, and
   getting back the transposed of the operation you asked for in C.

