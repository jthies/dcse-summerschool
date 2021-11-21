# Solution

The solution can be used to perform the numerical experiments from exercises 1-3 without coding.

## Numerical experiments - Exercise 1

Perform the following numerical experiments:

+ Try out different Krylov methods: change the parameter `"Belos Solver Type"` in the parameter list, e.g.,
  + `"CG"`
  + `"GMRES"`
  + `"BICGSTAB"`
+ Vary the problem size by changing the number of subdomains and the subdomain size. How does the iteration count depend on the problem size?

## Numerical experiments - Exercise 2

Perform the following numerical experiments:

+ How does the use of a one-level Schwarz preconditioner improve the convergence of the Krylov method? Compare the iteration counts against exercise 1.
+ Vary the width of the overlap using the parameter `"Overlap"`.
+ How does a variation of prolongation operator influence the convergence (parameter `"Combine Values in Overlap"`):
  + Standard additive Schwarz (`"Full"`)
  + Restricted additive Schwarz (`"Restricted"`)
  + Scaled additive Schwarz (`"Scaled"`)

## Numerical experiments - Exercise 3

Perform the following numerical experiments:

+ How does adding a second level improves the convergence compared to using a one-level Schwarz preconditioner? Compare against the convergence for the solution of exercise 2.

+ Compare against a reduced dimension GDSW (RGDSW) coarse space:

  ```xml
  <ParameterList name="InterfacePartitionOfUnity">
      <Parameter name="Type"             type="string" value="RGDSW"/>
  </ParameterList>
  ```
