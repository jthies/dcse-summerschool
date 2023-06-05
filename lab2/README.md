# Lab 2: Iterative Methods in Trilinos

In this lab we will solve a PDE problem using data structures and solvers from the Trilinos framework.
We will assess the suitability of various solvers and preconditioners for CPU and GPU computing.


## What you will practice here

- Compile and run MPI+X code (Modules, CMake, slurm, CPU and GPU nodes)
- Creating sparse linear systems with Trilinos
- Solving them using preconditioned Krylov methods
- Deciding which algorithms are suitable for which platform

## Your tasks

There are four individual exercises that take you from constructing a linear system
to standard iterative solvers and finally to more advanced preconditioners.
It is in principle possible to start with a later exercise. After you finish one of the
exercises, it may be fun to try running it on a GPU to check if you get the right result
and compare the performance to the CPU.

- [Creating and filling matrices and vectors](exercise-1)
- [Iterative solution of linear systems](exercise-2)
- [One-level Schwarz domain decomposition preconditioner using FROSch](exercise-3)
- [Two-level Schwarz domain decomposition preconditioner using FROSch](exercise-4)

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
