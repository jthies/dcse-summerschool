# Exercise 4 - Two-level Schwarz Domain Decomposition Preconditioner Using `FROSch`

In the previous three exercises, you have learned how to:

+ create and fill parallel distributed matrices and vectors,
+ solve a linear equation system using an iterative solver with a simple `Ifpack2` preconditioner,
+ use a one-level Schwarz domain decomposition preconditioners.

In this exercise, you will learn about the use of **two-level Schwarz preconditioners**, in particular, the **algebraic GDSW (Generlized Dryja-Smith-Widlund) preconditioners**. Therefore, you will construct the corresponding preconditioner objects using the [FROSch (Fast and Robust Overlapping Schwarz) package](https://shylu-frosch.github.io/) of Trilinos.

## Didactic goals

Upon completion of this exercise, you should be able to:

+ construct a GDSW preconditioner for a given matrix using the FROSch package.
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
  /* START OF TODO: FROSch::TwoLevelPreconditioner */
  
  
  
  /* END OF TODO: FROSch::TwoLevelPreconditioner */
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

1. Construct a two-level Schwarz preconditioner object using the matrix `A` and parameter list `precList` objects:

   ```c++
   RCP<twolevelpreconditioner_type> prec(new twolevelpreconditioner_type(A,precList));
   ```

   **Note:**

   + In the file `utils.hpp`, the type `TwoLevelPreconditioner_type` is defined:

     ```c++
     typedef FROSch::TwoLevelPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type> twolevelpreconditioner_type;
     ```

     So the actual type of the preconditioner is `FROSch::TwoLevelPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type>`. It has the same template parameters as a `Tpetra` matrix or vector.

2. Same as the one-level Schwarz preconditioner, the two-level Schwarz preconditioner with GDSW coarse space is set up using `initialize()` and `compute()`. For correctly setting up the GDSW coarse space, we have to specify the dimension of the problem (integer `dimension`), the number of degrees of freedom per node (integer `dofspernode`), and the overlap (parameter `"Overlap"` from the parameter list); the number of degrees of freedom per node depends on the dimension of the problem and the equation.

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

+ Can you confirm the condition number bound 

  <img src="https://render.githubusercontent.com/render/math?math=\kappa(M^{-1}K) \leq C (1%2B\frac{H}{\delta}) (1%2B\log(\frac{H}{h}))">, 

  where <img src="https://render.githubusercontent.com/render/math?math=C"> is a constant, <img src="https://render.githubusercontent.com/render/math?math=h"> is the element width, <img src="https://render.githubusercontent.com/render/math?math=H"> is the subdomain size, and <img src="https://render.githubusercontent.com/render/math?math=\delta"> is the width of the overlap? In order to do so, investigate the iteration counts of a preconditioner Krylov method.

+ Compare against a reduced dimension GDSW (RGDSW) coarse space:

  ```xml
  <ParameterList name="InterfacePartitionOfUnity">
      <Parameter name="Type"             type="string" value="RGDSW"/>
  </ParameterList>
  ```

## References

+ For more details on GDSW and RGDSW methods:

  ```latex
  @article {Dohrmann:2008:DDL,
      AUTHOR = {Dohrmann, Clark R. and Klawonn, Axel and Widlund, Olof B.},
       TITLE = {Domain decomposition for less regular subdomains: overlapping
                {S}chwarz in two dimensions},
     JOURNAL = {SIAM J. Numer. Anal.},
    FJOURNAL = {SIAM Journal on Numerical Analysis},
      VOLUME = {46},
        YEAR = {2008},
      NUMBER = {4},
       PAGES = {2153--2168},
        ISSN = {0036-1429},
     MRCLASS = {65N30 (65N55)},
    MRNUMBER = {2399412},
  MRREVIEWER = {Marius Ghergu},
         DOI = {10.1137/070685841},
         URL = {https://doi-org.tudelft.idm.oclc.org/10.1137/070685841},
  }
  
  @article {Dohrmann:2017,
      AUTHOR = {Dohrmann, Clark R. and Widlund, Olof B.},
       TITLE = {On the design of small coarse spaces for domain decomposition
                algorithms},
     JOURNAL = {SIAM J. Sci. Comput.},
    FJOURNAL = {SIAM Journal on Scientific Computing},
      VOLUME = {39},
        YEAR = {2017},
      NUMBER = {4},
       PAGES = {A1466--A1488},
        ISSN = {1064-8275},
     MRCLASS = {65N55 (65F08 65F10 65N30)},
    MRNUMBER = {3686806},
  MRREVIEWER = {Benjamin Wi-Lian Ong},
         DOI = {10.1137/17M1114272},
         URL = {https://doi-org.tudelft.idm.oclc.org/10.1137/17M1114272},
  }
+ For details on the parallel implementation:

  ```
  @article {Heinlein:2016:PIT,
      AUTHOR = {Heinlein, Alexander and Klawonn, Axel and Rheinbach, Oliver},
       TITLE = {A parallel implementation of a two-level overlapping {S}chwarz
                method with energy-minimizing coarse space based on
                {T}rilinos},
     JOURNAL = {SIAM J. Sci. Comput.},
    FJOURNAL = {SIAM Journal on Scientific Computing},
      VOLUME = {38},
        YEAR = {2016},
      NUMBER = {6},
       PAGES = {C713--C747},
        ISSN = {1064-8275},
     MRCLASS = {65F10 (65F08 65N55 65Y05)},
    MRNUMBER = {3579700},
  MRREVIEWER = {Olexander S. Babanin},
         DOI = {10.1137/16M1062843},
         URL = {https://doi-org.tudelft.idm.oclc.org/10.1137/16M1062843},
  }
  
  @incollection {Heinlein:2018:IPP,
      AUTHOR = {Heinlein, Alexander and Klawonn, Axel and Rheinbach, Oliver
                and Widlund, Olof B.},
       TITLE = {Improving the parallel performance of overlapping {S}chwarz
                methods by using a smaller energy minimizing coarse space},
   BOOKTITLE = {Domain decomposition methods in science and engineering
                {XXIV}},
      SERIES = {Lect. Notes Comput. Sci. Eng.},
      VOLUME = {125},
       PAGES = {383--392},
   PUBLISHER = {Springer, Cham},
        YEAR = {2018},
     MRCLASS = {65N55 (65Y05)},
    MRNUMBER = {3989887},
         DOI = {10.1007/978-3-319-93873-8\_3},
         URL = {https://doi-org.tudelft.idm.oclc.org/10.1007/978-3-319-93873-8_3},
  }
  ```

  
