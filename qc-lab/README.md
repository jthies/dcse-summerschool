# QC Lab: First steps on a Quantum Annealer

In this lab we will learn the first steps to solve combinatorial optimization problems on a quantum annealer. In particular, we will see how certain combinatorial optimization problems can be formulated as a Quadratic Unconstrained Binary Optimizaiton (QUBO) problem of the form
$$\min_{x\in\\{0,1\\}^n} x^\top Q x = \sum_{i=1}^n\sum_{j=i}^nQ_{ij}x_ix_j$$
with real-valued upper triangular matrix $Q\in\mathbb{R}^{n\times n}$

## What you will practice here
- setting up D-Wave's Ocean SDK and running a first QUBO for the MaxCut problem
- using D-Wave's problem inspector to 'debug' the computation
- developing a QUBO formulation for the traveling salesperson from scratch

## Your tasks

1. [Setup D-Wave Ocean SDK](exercise-1/Readme.md)
