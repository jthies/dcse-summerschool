# Source Files

+ [Exercise 1](https://github.com/searhein/frosch-demo/tree/main/src/exercise-1) - Implementing a Krylov Solver Using `Belos`
+ [Exercise 2](https://github.com/searhein/frosch-demo/tree/main/src/exercise-2) - Implementing a One-Level Schwarz Preconditioner Using `FROSch`
+ [Exercise 3](https://github.com/searhein/frosch-demo/tree/main/src/exercise-3) - Implementing a GDSW Preconditioner Using `FROSch`
+ [Solution](https://github.com/searhein/frosch-demo/tree/main/src/solution) contains an implementation that includes all the steps above.

## Description

All exercises are based on an implementation of a **simple Laplace or elasticity model problem in two or three dimensions**, which is already prepared for you in the `main.cpp` files; the equation and dimension of the problem can be selected via the input parameters. The computational domain is the **unit square or cube** in two or three dimensions, respectively.

![solution](https://github.com/searhein/frosch-demo/blob/main/images/solution.png?raw=true)

The discrete linear equation system is assembled using the Trilinos package `Galeri`; the Laplace equation is discretized using finite differences and the linear elasticity equation using finite elements.

The first part of each exercise is to **add the missing lines of code**, the second part is to **run some numerical experiments**. The solution can be used to perform the numerical experiments without coding.

## Additional Remarks

+ All the code changes have to be made within the respective `main.cpp` file in the subdirectory of each exercise: most of the code is already prepared, and it is sufficient to insert your code after the blocks
   ```
   /*
   ============================================================================
   !! INSERT CODE !!
   ----------------------------------------------------------------------------
   + ...
   ============================================================================
   */
   ```
   which also contain a short description fo the code that has to be inserted.

+ By running

   ```shell
   make
   ```

   within a subfolder of the `build` directory corresponding to one exercise/solution, only this exercise/solution will be compiled.

+ The programs can be executed as follows:

   ```shell
   mpirun -n 4 ./solution.exe [options]
   ```

   In this case, the test is run on 4 MPI ranks. Since FROSch **assumes a one-to-one correspondence of MPI ranks and subdomains**, the test automatically uses 4 subdomains. Moreover, the tests are based on a structured domain decomposition with

   + N^2 subdomains in two dimensions and
   + N^3 subdomains in three dimensions,

   for some N.

   ![subdomains](https://github.com/searhein/frosch-demo/blob/main/images/subdomains.png?raw=true)

+ The list of all options can be printed with:

   ```shell
   ./solution.exe --help
   ```

   This yields:

   ```shell
   Usage: ./solution.exe [options]
     options:
     --help                               Prints this help message
     --pause-for-debugging                Pauses for user input to allow attaching a debugger
     --echo-command-line                  Echo the command-line but continue as normal
     --dim                  int           Dimension: 2 or 3
                                          (default: --dim=2)
     --eq                   string        Type of problem: 'laplace' or 'elasticity'
                                          (default: --eq="laplace")
     --m                    int           Subdomain size: H/h (default 10)
                                          (default: --m=10)
     --prec                 string        Preconditioner type: 'none', '1lvl', or '2lvl'
                                          (default: --prec="1lvl")
     --xml                  string        File name of the parameter list (default ParameterList.xml).
                                          (default: --xml="parameters-2d.xml")
     --epetra               bool          Linear algebra framework: 'epetra' or 'tpetra' (default)
     --tpetra                             (default: --tpetra)
     --v                    int           Verbosity Level: VERB_DEFAULT=-1, VERB_NONE=0 (default), VERB_LOW=1, VERB_MEDIUM=2, VERB_HIGH=3, VERB_EXTREME=4
                                          (default: --v=0)
     --write                bool          Write VTK files of the partitioned solution: 'write' or 'no-write' (default)
     --no-write                           (default: --no-write)
     --timers               bool          Show timer overview: 'timers' or 'no-timers' (default)
     --no-timers                          (default: --no-timers)
   ```

   **Note:** *This is the full list of parameter for the solution code; some of the parameters are not valid for the previous exercises.*

+ Each exercise comes with one or two parameter list files for specifying the settings of the iterative solver and the preconditioner.

+ With the option `--write`, each MPI rank will write one file with its part of the solution. These files can be opened and visualized using [Paraview](https://www.paraview.org).

   **Note:** *A good way of visualizing the global solution is by loading all the files at once and using the filter `Group Datasets`*.

