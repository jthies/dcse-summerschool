# Exercise 2 - Implementing a One-Level Schwarz Preconditioner Using `FROSch`

Whereas, in exercise 1, we have implemented an iterative `Belos` solver for the linear system, we will now add a one-level Schwarz domain decomposition preconditioner to accelerate the convergence of the iterative solver. In particular, in part III of the code, we will construct and set up the preconditioner, and in part IV, we will specify it as the preconditioner for the iterative solver:

1. The one-level Schwarz preconditioner object is constructed using the matrix `A` and parameter list `precList` objects:

   ```c++
   RCP<onelevelpreconditioner_type> prec(new onelevelpreconditioner_type(A,precList));
   ```

   **Note:**

   + In the file `headers_and_helpers.hpp`, the type `onelevelpreconditioner_type` is defined:

     ```c++
     typedef FROSch::OneLevelPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type> onelevelpreconditioner_type;
     ```

2. The preconditioner is set up using `initialize()` and `compute()`. In the `initialize()` phase, the operations that depend on the structure of the matrix but are independent of matrix values are performed. In the `compute()` phase, the operations that depend on the matrix values are performed:

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

## Numerical experiments

Perform the following numerical experiments:

+ How does the use of a one-level Schwarz preconditioner improve the convergence of the Krylov method? Compare the iteration counts against exercise 1.
+ Vary the width of the overlap using the parameter `"Overlap"`.
+ How does a variation of prolongation operator influence the convergence (parameter `"Combine Values in Overlap"`):
  + Standard additive Schwarz (`"Full"`)
  + Restricted additive Schwarz (`"Restricted"`)
  + Scaled additive Schwarz (`"Scaled"`)
