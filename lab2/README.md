# Lab 2: Iterative Methods in Trilinos

In this lab we will solve a PDE problem using data structures and solvers from the Trilinos framework.
We will assess the suitability of various solvers and preconditioners for CPU and GPU computing.


## What you will practice here

- compile and run MPI+X code (Modules, CMake, slurm, CPU and GPU nodes)
- creating sparse linear systems with Trilinos
- solving them using preconditioned Krylov methods
- deciding which algorithms are suitable for which platform

## Your tasks

There are three individual exercises on
- [Creating and filling matrices and vectors](exercise-1)
- [Iterative solution of linear systems](exercise-2)
- [Using an advanced Domain-Decomposition preconditioner (FROSch)](exercise-3)

## Setup on DelftBlue

- To load the required modules and set environment variables, use ``source env.sh`` in the base directory ``lab2/``. This script behaves differently on GPU nodes, so we included the command in the slurm scripts provided.
- Compiling for the GPU nodes should be done on the GPU nodes.
- Trilinos uses CMake to create makefiles based on your environment. To configure and compile, e.g. exampole-1:
```bash
source env.sh
cd example-1
mkdir build
cd build
cmake ..
make
```
You can then use the ``build`` folder for running your experiments.
Very small test runs can be done on the login node, but for larger runs please submit a job.
A sample job script for the CPU and GPU experiments is provided for the first exercise and can be adapted for the others.
