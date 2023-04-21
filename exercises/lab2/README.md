# Lab 2: Iterative Methods in Trilinos

Given a code framework, students will

Perhaps we can use some more interesting FE application here than just Poisson again (in particular something non-symmetric)?

- compile and run MPI+X code (Modules, CMake, slurm)
- create a sparse matrix and fill it with the coefficients for a (non-symmetric) PDE problem
- solve it using various Krylov methods: look at timings for GMRES, GMRES(m) vs. short recurrence method like BiCGStab
- try out some preconditioning techniques, observe node-level performance of ILU vs. a polynomial method
- run the same experiments on CPU and GPU nodes with Kokkos/Ifpack2 etc.
- set up a one- and two-level Additive Schwarz preconditioner, experiment with different ways of splitting up the domain (CPU only)
