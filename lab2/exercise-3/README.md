# Exercise 3 - One-level Schwarz Domain Decomposition Preconditioner Using `FROSch`

In the previous two exercises, you have learned how to:

+ create and fill parallel distributed matrices and vectors and
+ solve a linear equation system using an iterative solver with a simple `Ifpack2` preconditioner.

In this exercise and the [next exercise](../exercise-4), you will learn about the use of more **advanced preconditioning techniques based on domain decomposition methods**. Therefore, you will construct preconditioner objects using the [FROSch (Fast and Robust Overlapping Schwarz) package](https://shylu-frosch.github.io/) of Trilinos:

+ **This exercise:** One-level Schwarz preconditioner
+ **[Next exercise](../exercise-4):** Two-level Schwarz preconditioner

## Didactic goals

Upon completion of this exercise, you should be able to:

+ construct a one-level Schwarz preconditioner for a given matrix using the FROSch package.
+ employ a FROSch preconditioner object as a preconditioner for a `Belos` iterative solver.
+ change the preconditioner settings via the parameter list.

 ## Base code

The base code for this exercise, which is already prepared for you in the `main.cpp` file, implements a **simple Laplace or elasticity model problem in two or three dimensions**; the equation and dimension of the problem can be selected via the input parameters. The computational domain is the **unit square or cube** in two or three dimensions, respectively.

![solution](https://github.com/searhein/frosch-demo/blob/main/images/solution.png?raw=true)

The discrete linear equation system is assembled using the Trilinos package `Galeri`; the Laplace equation is discretized using finite differences and the linear elasticity equation using finite elements.

Without any code changes, the model problem is solved using an iterative `Belos` solver without any preconditioner.

## Instructions

The exercise consists of two parts:

+ **Implementation:** add missing lines of code. The corresponding locations are marked by

   ```
   /* START OF TODO: FROSch::OneLevelPreconditioner */



   /* END OF TODO: FROSch::OneLevelPreconditioner */
   ```

+ **Numerical experiments:** run numerical experiments to test the preconditioner. The programs can be executed as follows:

   ```shell
   mpirun -n 4 ./solution.exe [options]
   ```

   In this case, the test is run on 4 MPI ranks. Since FROSch **assumes a one-to-one correspondence of MPI ranks and subdomains**, the test automatically uses 4 subdomains. Moreover, the tests are based on a structured domain decomposition with

   + N^2 subdomains in two dimensions and
   + N^3 subdomains in three dimensions,

   for some N.

   ![subdomains](https://github.com/searhein/frosch-demo/blob/main/images/subdomains.png?raw=true)

   The list of all options can be printed with:

   ```shell
   ./main.x --help
   ```

   Other settings may be changed by modifying the parameter list files `parameters-2d.xml` and `parameters-3d.xml` for the 2D and 3D cases, respectively.

### Implementation

1. Construct a one-level Schwarz preconditioner object using the matrix `A` and parameter list `precList` objects:

   ```c++
   RCP<onelevelpreconditioner_type> prec(new onelevelpreconditioner_type(A,precList));
   ```

   **Note:**

   + In the file `utils.hpp`, the type `onelevelpreconditioner_type` is defined:

     ```c++
     typedef FROSch::OneLevelPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type> onelevelpreconditioner_type;
     ```

     So the actual type of the preconditioner is `FROSch::OneLevelPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type>`. It has the same template parameters as a `Tpetra` matrix or vector.

2. The preconditioner is set up using `initialize()` and `compute()`. In the `initialize()` phase, the **operations that depend on the structure of the matrix but are independent of matrix values** are performed; this is similar to the symbolic factorization of a direct solver. In the `compute()` phase, the **operations that depend on the matrix values** are performed:

   ```c++
   prec->initialize(false);
   prec->compute();
   ```

   **Note:**

   + The input parameter `false ` of `initialize()` ensures that the width of the overlap is read from the parameter list, i.e., the parameter `"Overlap"`.

3. In order to be used as a preconditioner for the `Belos` iterative solver, the preconditioner object has to wrapped as a `Belos::OperatorT<multivector_type>` using:

   ```c++
   RCP<operatort_type> belosPrec = rcp(new xpetraop_type(prec));
   ```

4. The `FROSch` preconditioner is then specified as the preconditioner for the `Belos` iteration using (already in the code):

   ```c++
   linear_problem->setRightPrec(belosPrec);
   ```

### Numerical experiments

Perform the following numerical experiments:

+ How does the use of a one-level Schwarz preconditioner improve the convergence of the Krylov method? Compare the iteration counts against the corresponding unpreconditioned iteration.

+ Can you confirm the condition number bound

  <img src="https://render.githubusercontent.com/render/math?math=\kappa(M^{-1}K) \leq C(1%2B\frac{1}{H \delta}))">,

  where <img src="https://render.githubusercontent.com/render/math?math=C"> is a constant, <img src="https://render.githubusercontent.com/render/math?math=H"> is the subdomain size, and <img src="https://render.githubusercontent.com/render/math?math=\delta"> is the width of the overlap? In order to do so, investigate the iteration counts of a preconditioner Krylov method.

+ How does a variation of the prolongation operator influence the convergence (parameter `"Combine Values in Overlap"`):
  + Standard additive Schwarz (`"Full"`): <img src="https://render.githubusercontent.com/render/math?math=M_{\rm OS-1}^{-1} K = \sum_{i=1}^N R_i^T K_i^{-1} R_i K">,
  + Restricted additive Schwarz (`"Restricted"`): <img src="https://render.githubusercontent.com/render/math?math=M_{\rm OS-1}^{-1} K = \sum_{i=1}^N \tilde  R_i^T K_i^{-1} R_i K"> with<img src="https://render.githubusercontent.com/render/math?math=\sum_{i=1} \tilde R_i^T R_i = I"> by defining the prolongation <img src="https://render.githubusercontent.com/render/math?math=\tilde R_i^T"> based on a unique partition.
  + Scaled additive Schwarz (`"Scaled"`): <img src="https://render.githubusercontent.com/render/math?math=M_{\rm OS-1}^{-1} K = \sum_{i=1}^N \tilde  R_i^T K_i^{-1} R_i K"> with<img src="https://render.githubusercontent.com/render/math?math=\sum_{i=1} \tilde R_i^T R_i = I"> by defining the prolongation <img src="https://render.githubusercontent.com/render/math?math=\tilde R_i^T"> by scaling <img src="https://render.githubusercontent.com/render/math?math=R_i^T"> with the inverse multiplicity.

  For more details on **restricted additive Schwarz preconditioners**, see:

  ```latex
  @article {Cai:1999:RAS,
      AUTHOR = {Cai, Xiao-Chuan and Sarkis, Marcus},
       TITLE = {A restricted additive {S}chwarz preconditioner for general
                sparse linear systems},
     JOURNAL = {SIAM J. Sci. Comput.},
    FJOURNAL = {SIAM Journal on Scientific Computing},
      VOLUME = {21},
        YEAR = {1999},
      NUMBER = {2},
       PAGES = {792--797},
        ISSN = {1064-8275},
     MRCLASS = {65N30 (65F35)},
    MRNUMBER = {1718707},
         DOI = {10.1137/S106482759732678X},
         URL = {https://doi-org.tudelft.idm.oclc.org/10.1137/S106482759732678X},
  }
  ```
