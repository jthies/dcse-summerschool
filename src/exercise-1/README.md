# Exercise 1 - Implementing a Krylov Solver Using `Belos`

In parts I and II of the prepared code, the system matrix as well as a right hand side and solution vector are constructed. The goal of this exercise is to implement an iterative solver for the linear system using the package `Belos` in part IV of the code:

1. The first step is to set up a `Belos::LinearProblem<scalar_type,multivector_type,operatort_type>` with the system matrix `A` or the `Belos::OperatorT` wrapper `belosA`, respectively, solution vector `x`, and right hand side `b`. 

   ```c++
   RCP<linear_problem_type> linear_problem (new linear_problem_type(belosA,x,b));
   linear_problem->setProblem(x,b);
   ```

   **Notes:** 

   + In the file `headers_and_helpers.hpp`, the type `linear_problem_type` are defined:

     ```c++
     typedef Belos::LinearProblem<scalar_type,multivector_type,operatort_type> linear_problem_type;
     ```

   + `RCP` is a smart pointer from the package `Teuchos`

2. Next, a `Belos::SolverManager<scalar_type,multivector_type,operatort_type> solver_type` object, which will call the iteration, has to be constructed using a `Belos::SolverFactory<scalar_type,multivector_type,operatort_type> solverfactory_type`. In particular:

   ```c++
   solverfactory_type solverfactory;
   RCP<solver_type> solver = solverfactory.create(parameterList->get("Belos Solver Type","GMRES"),belosList);
   ```

   Then, the linear equation system has to be specified and the iteration called:

   ```c++
   solver->setProblem(linear_problem);
   solver->solve();
   ```

   **Notes:** 

   + In the file `headers_and_helpers.hpp`, the types `solverfactory_type`  and `solver_type` are defined:

     ```c++
     typedef Belos::SolverFactory<scalar_type,multivector_type,operatort_type> solverfactory_type;
     typedef Belos::SolverManager<scalar_type,multivector_type,operatort_type> solver_type;
     ```

**Numerical experiments:**

+ Try out different Krylov methods: change the parameter `"Belos Solver Type"` in the parameter list, e.g.,
  + `"CG"`
  + `"GMRES"`
  + `"BICGSTAB"`
+ Vary the problem size by changing the number of subdomains and the subdomain size. How does the iteration count depend on the problem size?
