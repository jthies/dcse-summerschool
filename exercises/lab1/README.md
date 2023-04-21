# Lab 1: Dense LU decomposition with OpenMP tasks

(Introductory text)

## What you will practice here

- Working with BLAS3 kernels
- Identifying data dependencies
- Task-based programming using OpenMP

## Your tasks

1. Compile and run the sequential code for different problem sizes. What is the computational complexity according to the measured runtimes?
2. Insert OpenMP pragma's to achieve task-parallel factorization of each tile. Measure the scalability.
3. Can you achieve task-based parallelization across tiles? Measure the scalability again
4. Measure key performnace metrics on up to 48 cores
