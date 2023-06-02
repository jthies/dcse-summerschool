# Analyzing SpMV performance using Likwid

In the lecture this morning, you learned about the Sparse Matrix-Vector (SpMV) multiplication, and its
performance characteristics. in this lab, we will try out the Trilinos/Tpetra implementation of this key component
of all sparse iterative solvers and investigate its performance using the tools of the ``likwid`` suite.

## What you will practice here

- applying the Roofline model
- getting information from performance counters with likwid
- relating the gathered data back to the model

# Setup on DelftBlue

The exercise is based on Trilinos, like lab2.
- As before, use ``source env.sh`` to setup the environment.
- Run ``./getSuiteSparseMatrices.sh`` to download some interesting sparse matrices from Gerhard Welleins lecture.
They will be extracted in your scratch space under /scratch/$USER/suite-sparse-matrices/

Using likwid requires certain settings that are only available on the reserved nodes. The measurements
concern multi-core CPUs: you may run the benchmarks without likwid on a GPU for fun and see how that works out
(adapt a the ``compile_and_run_on_gpu.slurm`` script from lab2).

# Your tasks

1. For at least one of the matrices, estimate the data traffic required for an SpMV as:
    - loading one double (value) and one integer (column index) per matrix element
    - loading one integer per row (row pointer)
    - loading once the input vector `x`
    - loading and storing once the result vector `y=Ax.`
1. Run the driver with this matrix, measuring the MEM_DP group with ``likwid-perfctr.``
   Use four MPI processes with 12 threads each.
   This is set up for you in the job script. Submit it with, e.g.: ``sbatch run_spmv_on_cpu.sh /scratch/$USER/suite-sparse-matrices/af_shell10.mtx``.
   Assuming a memory bandwidth of about 200 GB/s on the complete node (48 threads), do your predictions match the measurement?
2. Repeat the experiment with one MPI process and 48 threads.
