# Exercise 3 - Implementing a GDSW Preconditioner Using `FROSch`

In exercise 3, instead of using a one-level Schwarz preconditioner to accelerate the convergence of the iterative solver as in exercise 2, we use a two-level Schwarz GDSW preconditioner:

1. The one-level Schwarz preconditioner object is constructed using the matrix `A` and parameter list `precList` objects:

   ```c++
   RCP<twolevelpreconditioner_type> prec(new twolevelpreconditioner_type(A,precList));
   ```

   **Note:**

   + In the file `headers_and_helpers.hpp`, the type `TwoLevelPreconditioner_type` is defined:

     ```c++
     typedef FROSch::TwoLevelPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type> twolevelpreconditioner_type;
     ```

2. As the one-level Schwarz preconditioner, the two-level Schwarz preconditioner with GDSW coarse space is set up using `initialize()` and `compute()`. For correctly setting up the GDSW coarse space, we have to specify the dimension of the problem (integer `dimension`), the number of degrees of freedom per node (integer `dofspernode`), and the overlap (parameter `"Overlap"` from the parameter list); the number of degrees of freedom per node depends on the dimension of the problem and the equation.

   ```c++
   prec->initialize(dimension,dofspernode,precList->get("Overlap",1));
   prec->compute();
   ```

   **Note:**

   + The parameters `dimension` and `dofspernode` could also be read directly from the parameter list by `FROSch`, however, this would require adjusting the parameter list every time the dimension or equation is changed. Therefore, we opt for using them as input parameters for `initialize()`.

   + Scalability for linear elasticity additionally requires rotational basis functions. In order to construct them, we have to set the parameter `"Null Space Type"` to `"Lineare Elasticity"`. This can be done within the code:

     ```c++
     precList->set("Null Space Type","Linear Elasticity");
     ```

     Additionally, we have to provide the coordinates of the nodes:

     ```c++
     prec->initialize(dimension,dofspernode,precList->get("Overlap",1),null,coordinates);
     ```



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

+ How does adding a second level improves the convergence compared to using a one-level Schwarz preconditioner? Compare against the convergence for the solution of exercise 2.

+ Compare against a reduced dimension GDSW (RGDSW) coarse space:

  ```xml
  <ParameterList name="InterfacePartitionOfUnity">
      <Parameter name="Type"             type="string" value="RGDSW"/>
  </ParameterList>
  ```
