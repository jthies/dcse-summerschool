# Solution

The solution can be used to perform the numerical experiments from exercises 1-3 without coding.

## Numerical experiments - Exercise 1

Perform the following numerical experiments:

+ Try out different Krylov methods: change the parameter `"Belos Solver Type"` in the parameter list, e.g.,
  + `"CG"`
  + `"GMRES"`
  + `"BICGSTAB"`
+ The condition number of the system matrix is bounded as follows:

  <img src="https://render.githubusercontent.com/render/math?math=\kappa(K) \leq \frac{C}{h^2}">, 
  where <img src="https://render.githubusercontent.com/render/math?math=C"> is a constant, and <img src="https://render.githubusercontent.com/render/math?math=h"> is the element width. Can you confirm that the iteration count grows when the mesh is refined?

+ Vary the problem size by changing the number of subdomains and the subdomain size. How does the iteration count depend on the problem size?

## Numerical experiments - Exercise 2

Perform the following numerical experiments:

+ How does the use of a one-level Schwarz preconditioner improve the convergence of the Krylov method? Compare the iteration counts against exercise 1.
+ Can you confirm the condition number bound 
  <img src="https://render.githubusercontent.com/render/math?math=\kappa(M^{-1}K) \leq C(1%2B\frac{1}{H \delta}))">, 
  where <img src="https://render.githubusercontent.com/render/math?math=C"> is a constant, <img src="https://render.githubusercontent.com/render/math?math=H"> is the subdomain size, and <img src="https://render.githubusercontent.com/render/math?math=\delta"> is the width of the overlap? In order to do so, investigate the iteration counts of a preconditioner Krylov method.
+ Vary the width of the overlap using the parameter `"Overlap"`.
+ How does a variation of prolongation operator influence the convergence (parameter `"Combine Values in Overlap"`):
  + Standard additive Schwarz (`"Full"`): <img src="https://render.githubusercontent.com/render/math?math=M_{\rm OS-1}^{-1} K = \sum_{i=1}^N R_i^T K_i^{-1} R_i K">,
  + Restricted additive Schwarz (`"Restricted"`): <img src="https://render.githubusercontent.com/render/math?math=M_{\rm OS-1}^{-1} K = \sum_{i=1}^N \tilde  R_i^T K_i^{-1} R_i K"> with<img src="https://render.githubusercontent.com/render/math?math=\sum_{i=1} \tilde R_i^T R_i = I"> by defining the prolongation <img src="https://render.githubusercontent.com/render/math?math=\tilde R_i^T"> based on a unique partition.
  + Scaled additive Schwarz (`"Scaled"`): <img src="https://render.githubusercontent.com/render/math?math=M_{\rm OS-1}^{-1} K = \sum_{i=1}^N \tilde  R_i^T K_i^{-1} R_i K"> with<img src="https://render.githubusercontent.com/render/math?math=\sum_{i=1} \tilde R_i^T R_i = I"> by defining the prolongation <img src="https://render.githubusercontent.com/render/math?math=\tilde R_i^T"> by scaling <img src="https://render.githubusercontent.com/render/math?math=R_i^T"> with the inverse multiplicity.

## Numerical experiments - Exercise 3

Perform the following numerical experiments:

+ How does adding a second level improves the convergence compared to using a one-level Schwarz preconditioner? Compare against the convergence for the solution of exercise 2.

+ Can you confirm the condition number bound 
  <img src="https://render.githubusercontent.com/render/math?math=\kappa(M^{-1}K) \leq C (1%2B\frac{H}{\delta}) (1%2B\log(\frac{H}{h}))">, 
  where <img src="https://render.githubusercontent.com/render/math?math=C"> is a constant, <img src="https://render.githubusercontent.com/render/math?math=h"> is the element width, <img src="https://render.githubusercontent.com/render/math?math=H"> is the subdomain size, and <img src="https://render.githubusercontent.com/render/math?math=\delta"> is the width of the overlap? In order to do so, investigate the iteration counts of a preconditioner Krylov method.

+ Compare against a reduced dimension GDSW (RGDSW) coarse space:

  ```xml
  <ParameterList name="InterfacePartitionOfUnity">
      <Parameter name="Type"             type="string" value="RGDSW"/>
  </ParameterList>
  ```
