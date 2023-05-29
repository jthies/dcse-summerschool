# Lab 4: Performance optimization of communication-avoiding algorithms

In this lab, we will construct and test various Krylov variants with reduced impact of orthogonalization
("communication-avoiding" methods. The linear system is given as Matrix Market input files and comes from
the solution of the momentum equations in a wind-turbine model (a non-symmetric problem: see the ``MomentumEQS*`` files in the directory
``/beegfs/apps/unsupported/summerschool/``,
which you need to unzip once using ``gunzip <filename>``). While this may seem an easy system to solve with a Krylov method,
note that solving thousands of such systems in sequence is required for a time-dependent simulation of this problem, so performance
**really matters**.

## What you will practice here

- working with Trilinos Belos/Tpetra/Ifpack2
- tuning solver parameters
- performance analysis using likwid

## Your task

Choose any of the following methods to investigate. Many are available directly in the given ifpack_driver code by
choosing the right input arguments. Collaborate with your classmates to try to solve the linear problem
as fast as possible on a DelftBlue compute node and/or a V100s GPU. Make the comparison fair and transparent by
 measuring the time for setup of the preconditioner and running the solver separately.

1. s-step GMRES after left scaling of the linear problem (i.e.: Jacobi(1) preconditioning) (available as a solver variant in the ifpack_driver).
2. GMRES preconditioned by **s Jacobi iterations** (see the "RELAXATION" preconditioner option in the ifpack_driver)
3. GMRES preconditioned by a degree-s **GMRES polynomial**. See [this Belos class](https://docs.trilinos.org/dev/packages/belos/doc/html/classBelos_1_1GmresPolySolMgr.html#details).
4. The IDR(s) method provided as a Fortran code here (also supports polynomial preconditioning)
5. Your own choice: e.g., other Krylov methods, a Domain Decomposition preconditioner, a Chebyshev polynomial preconditioner, etc.

## Guiding your choices using builtin profiling and likwid

Play with varying values of the polynomial degree s, and see how the balance between e.g. SpMV's, preconditioner application and orthogonalization costs changes
(these are reported separately by Belos).
We have instrumented the Trilinos code sections timed by Belos using ``likwid`` markers.
The command
```
likwid-perfctr -m -C 0-47 -g MEM_DP <program> <args>
```
can be used to read out the most relevant performnace group and report computational intensity.
Observe, e.g., how the computational intensity of the s-step GMRES method increases with s, and how
the orthogonalization cost increases with the number of (outer) GMRES iterations needed.
