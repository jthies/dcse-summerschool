# DCSE summerschool 2023 slides and exercise material

This repository contains slides and lab assignments for the summerschool,
it will be available for participants during and after the event.

## Structure

Each subdimrectory in the 'exercises' has a README.md with self-contained instructions.
We will generate a print-out from those files for each exercise.

## Overview of the lab sessions

### Lab 1 (monday moring): dense linear algebra and data dependencies

Dense LU with OpenMP tasks.

- Implement and benchmark three variants: tasking along rows/cols/diagonal

### GPU lab (monday afternoon)

GPUs for linear algebra

- managing memory between host and device
- write first CUDA kernels (initialize matrix, GEMV)
- Performance of GEMV vs GEMM, compare with CuBLAS and roofline model 
- understand square vs. tall&skinny matrices
- Use nvprof to measure and understand performance

### Lab 2 (tuesday): Sparse solvers, Trilinos solvers and preconditioners

- based on Trilinos tutorial
- understand how to work with sparse matrices and maps: create, repartition and reorder linear system
- try different solver/precond combinations
- find out experimentally which algorithms are suitable for CPUs and GPUs

## Lab 3 (wednesday): performance of SpMV in Trilinos

Short lab: measure data traffic in Krylov solver using likwid
for some different matrices -> anything wrong?

## Lab 4 (thursday): communication-avoiding Krylov methods

- Run s-step GMRES in Trilinos, investigate performance of matrix powers kernel and orthogonalization
  using likwid (-> matrix power s has s times the data traffic, ortho becomes more compute intensive with larger s)
- Run same experiments on GPU and compare timing behavior depending on s.
        -

