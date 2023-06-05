This tutorial was given at the [EuroTUG](https://github.com/EuroTUG/trilinos-docker)ex02


# How to solve a linear system with Belos and Ifpack2?

In this exercise, you will use GMRES (potentially with a preconditioner such as Jacobi/Gauss-Seidel/Symmetric Gauss-Seidel) to solve linear systems.

## Didactic goals

This exercise will introduce you to the basics of preconditioned Krylov solvers in Trilinos. Therefore, you will work with Krylov solvers from the `Belos` package and relaxation methods from the `Ifpack2` package to be used as preconditioners.

You will learn

- how to create and configure a Krylov solver via `Belos,`
- how to create and configure a relaxation method via `Ifpack2,`
- how to use the relaxation method as a preconditioner for the outer Krylov method.
- how to setup and solve the linear system.

## Complete the source code to create and use the Krylov solver from `Belos`

As in exercise 1, fill in the missing lines of code in the locations marked by ``TODO`` comments.

For now, you will use GMRES only without any preconditioner. In the code, you will skip all parts related to preconditioning for now and will come back to them later.

1. At `/* START OF TODO: Create Belos solver */`, use the `Belos::SolverFactory` to create a GMRES solver:

   ```cpp
   Belos::SolverFactory<scalar_type, multivec_type, operator_type> belosFactory;
   solver = belosFactory.create ("GMRES", solverParams);
   ```

1. At `/* START OF TODO: Define linear problem */`, define the linear system A*x=b:

   ```c++
   problem = rcp(new problem_type (matrix, x, rhs));
   ```

1. At `/* START OF TODO: Set the linear problem */`, pass the linear problem to the Krylov solver:

   ```cpp
   problem->setProblem();
   solver->setProblem(problem);
   ```

1. At `/* START OF TODO: Solve */`, call the solver and capture the solver's convergence status:

   ```cpp
   Belos::ReturnType solveResult = solver->solve();
   ```

## Build and run the app

1. Compile the app via `make`.

Now, your can execute your application (via a slurm job or -- for very small tests -- on the login node) and try various linear systems and solver settings.
You can configure the behavior on the command line.
Here's an overview of the configuration options:

- You can choose between different matrices (`Laplace1D`, `Laplace2D`, `Laplace3D`, `Elasticity2D`, `Elasticity3D`, ...).
- You can define the mesh size by specifying the number of mesh nodes `nx`, `ny`, `nz` in x-, y-, z-direction, respectively.
- You can set the max number of iterations (`maxIters`), the desired tolerance (`tol`), and the output frequency (`outFrequency`)

To get a complete list of options, run `./main.x --help`.

1. Run the solver with default settings: `./main.x`

   Expected output:

   ```
   >> I. Create linear system A*x=b for a Laplace2D problem.
   >> II. Create a GMRES solver from the Belos package.
   >> III. Solve the linear system.
   Passed.......OR Combination ->
     OK...........Number of Iterations = 20 < 100
     Converged....(2-Norm Res Vec) / (2-Norm Prec Res0)
                  residual [ 0 ] = 7.70474e-05 < 0.0001

   Belos converged in 20 iterations to an achieved tolerance of 7.70474e-05 (< tol = 0.0001).
   ```

1. Run the solver with a tighter tolerance of 1.0e-8: `./main.x --tol=1.0e-8`

   Expected output:

   ```
   >> I. Create linear system A*x=b for a Laplace2D problem.
   >> II. Create a GMRES solver from the Belos package.
   >> III. Solve the linear system.
   Passed.......OR Combination ->
     OK...........Number of Iterations = 32 < 100
     Converged....(2-Norm Res Vec) / (2-Norm Prec Res0)
                  residual [ 0 ] = 8.66916e-09 < 1e-08

   Belos converged in 32 iterations to an achieved tolerance of 8.66916e-09 (< tol = 1e-08).
   ```

1. Run the solver in parallel on 4 MPI ranks to solve an `Elasticity3D` problem on a 25x12x30 mesh up to a relative tolerance of `tol` = 1.0e-8 and allow 400 Krylov iterations at maximum: `mpirun -np 4 ./ex_03_solve --tol=1.0e-8  --maxIters=400 --matrixType=Elasticity3D --nx=25 --ny=12 --nz=30`

   Expected output:

   ```
   >> I. Create linear system A*x=b for a Elasticity3D problem.
   >> II. Create a GMRES solver from the Belos package.
   >> III. Solve the linear system.
   Passed.......OR Combination ->
     OK...........Number of Iterations = 377 < 400
     Converged....(2-Norm Res Vec) / (2-Norm Prec Res0)
                  residual [ 0 ] = 9.70154e-09 < 1e-08
   
   Belos converged in 377 iterations to an achieved tolerance of 9.70154e-09 (< tol = 1e-08).
   ```

Now, feel free to experiment with some solver settings, mesh sizes, etc.

## Create and use a preconditioner

1. At `/* START OF TODO: Create preconditioner */`, ask the `Ifpack2::Factory` to create a relaxation preconditioner:

   ```cpp
   prec = Ifpack2::Factory::create<row_matrix_type> ("RELAXATION", matrix);
   ```

1. At `/* START OF TODO: Configure preconditioner */`, collect the preconditioner settings in a `Teuchos::ParameterList`
and pass it to the preconditioner:

   ```cpp
   ParameterList precParams;
   precParams.set("relaxation: type", relaxationType);
   precParams.set("relaxation: sweeps", numSweeps);
   precParams.set("relaxation: damping factor", damping);
   prec->setParameters(precParams);
   ```

1. At `/*START OF TODO: Setup the preconditioner*/`, initialize and compute the preconditioner such that it is ready to be used:

   ```cpp
   prec->initialize();
   prec->compute();
   ```

1. At `/* START OF TODO: Insert preconditioner */`, tell the linear problem about the existence of the preconditioner, such that it can then be called from the Krylov solver:

   ```cpp
   problem->setRightPrec(prec);
   ```

Now, you can run examples with a _preconditioned_ Krylov solver.
To study the effect of the preconditioner, we look at the last example from the unpreconditioned GMRES
and examine different preconditioner settings.
To enable the preconditioer, just pass the additional command line argument `--withPreconditioner`.

1. Run the solver in parallel on 4 MPI ranks to solve an `Elasticity3D` problem on a 25x12x30 mesh up to a relative tolerance of `tol` = 1.0e-8 and allow 400 Krylov iterations at maximum and enable the default preconditioner: `mpirun -np 4 ./ex_03_solve --tol=1.0e-8  --maxIters=400 --matrixType=Elasticity3D --nx=25 --ny=12 --nz=30 --withPreconditioner`

   Expected output:

   ```
   >> I. Create linear system A*x=b for a Elasticity3D problem.
   >> II. Create a preconditioned GMRES solver from the Belos package.
   >> III. Solve the linear system.
   Passed.......OR Combination ->
     OK...........Number of Iterations = 294 < 400
     Converged....(2-Norm Res Vec) / (2-Norm Prec Res0)
                  residual [ 0 ] = 9.46091e-09 < 1e-08

   Belos converged in 294 iterations to an achieved tolerance of 9.46091e-09 (< tol = 1e-08).
   ```

   Through the preconditioner (in this case 1x sweep of damped Jacobi (damping = 2/3) per GMRES iteration),
   the total number of iterations drops from 377 to 294.

1. Run the solver in parallel on 4 MPI ranks to solve a `Elasticity3D` problem on a 25x12x30 mesh up to a relative tolerance of `tol` = 1.0e-8 and allow 400 Krylov iterations at maximum and enable 2 sweeps of the default preconditioner: `mpirun -np 4 ./ex_03_solve --tol=1.0e-8  --maxIters=400 --matrixType=Elasticity3D --nx=25 --ny=12 --nz=30 --withPreconditioner --numSweeps=2`

   Expected output:

   ```
   >> I. Create linear system A*x=b for a Elasticity3D problem.
   >> II. Create a preconditioned GMRES solver from the Belos package.
   >> III. Solve the linear system.
   Passed.......OR Combination ->
     OK...........Number of Iterations = 154 < 400
     Converged....(2-Norm Res Vec) / (2-Norm Prec Res0)
                  residual [ 0 ] = 9.90015e-09 < 1e-08

   Belos converged in 154 iterations to an achieved tolerance of 9.90015e-09 (< tol = 1e-08).
   ```

   Through the preconditioner (in this case 1x sweep of damped Jacobi (damping = 2/3) per GMRES iteration),
   the total number of iterations drops from 377 to 294.

1. Run the solver in parallel on 4 MPI ranks to solve an `Elasticity3D` problem on a 25x12x30 mesh up to a relative tolerance of `tol` = 1.0e-8 and allow 400 Krylov iterations at maximum and use 1 sweep of damped Gauss-Seidel as preconditioner: `mpirun -np 4 ./ex_03_solve --tol=1.0e-8  --maxIters=400 --matrixType=Elasticity3D --nx=25 --ny=12 --nz=30 --withPreconditioner --precType=Gauss-Seidel`

   Expected output:

   ```
   >> I. Create linear system A*x=b for a Elasticity3D problem.
   >> II. Create a preconditioned GMRES solver from the Belos package.
   >> III. Solve the linear system.
   Passed.......OR Combination ->
     OK...........Number of Iterations = 248 < 400
     Converged....(2-Norm Res Vec) / (2-Norm Prec Res0)
                  residual [ 0 ] = 9.49484e-09 < 1e-08

   Belos converged in 248 iterations to an achieved tolerance of 9.49484e-09 (< tol = 1e-08).
   ```

1. Run the solver in parallel on 4 MPI ranks to solve an `Elasticity3D` problem on a 25x12x30 mesh up to a relative tolerance of `tol` = 1.0e-8 and allow 400 Krylov iterations at maximum and use 2 sweeps of damped Symmetric Gauss-Seidel as preconditioner: `mpirun -np 4 ./ex_03_solve --tol=1.0e-8  --maxIters=400 --matrixType=Elasticity3D --nx=25 --ny=12 --nz=30 --withPreconditioner --precType="Symmetric Gauss-Seidel" --numSweeps=2`

   Expected output:

   ```
   >> I. Create linear system A*x=b for a Elasticity3D problem.
   >> II. Create a preconditioned GMRES solver from the Belos package.
   >> III. Solve the linear system.
   Passed.......OR Combination ->
     OK...........Number of Iterations = 107 < 400
     Converged....(2-Norm Res Vec) / (2-Norm Prec Res0)
                  residual [ 0 ] = 8.79942e-09 < 1e-08
   
   Belos converged in 107 iterations to an achieved tolerance of 8.79942e-09 (< tol = 1e-08).
   ```

As expected, the number of Krylov iterations until convergence goes down,
when a stronger preconditioner is used.

Now, feel free to experiment with other solver and preconditioner configurations, mesh sizes, etc.
