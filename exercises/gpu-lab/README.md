# GPU lab: Dense Matrix-vector or CG

(Introductory text)

- Problem 1: Lanczos method for computing the smallest eigenvalue of a dense matrix
- Problem 2: Same for 2D version that can be written as (I (x) C) + (C (x) I)

# Your tasks


- what is the computational intensity of the operation (in Flops/Byte)?
- implement GEMV using CUDA, compare performance with CUBLAS and performance model
- implement axpy and dot, run simple tests and benchmarks
- take a Lanczos code, do memory allocation using CUDA managed memory, compute largest eigenvalue of a dense matrix
- compare overall performance CPU/GPu
- Extend your operator to a 2D problem of the form (I (x) C) + (C (x) I) and implement it using GEMM, compare again CPU/GPU.
