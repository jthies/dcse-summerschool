# Source Files

+ [Exercise 1](https://github.com/searhein/frosch-demo/tree/main/src/exercise-1) - Implementing a Krylov Solver Using `Belos`
+ [Exercise 2](https://github.com/searhein/frosch-demo/tree/main/src/exercise-2) - Implementing a One-Level Schwarz Preconditioner Using `FROSch`
+ [Exercise 3](https://github.com/searhein/frosch-demo/tree/main/src/exercise-3) - Implementing a GDSW Preconditioner Using `FROSch`
+ [Solution](https://github.com/searhein/frosch-demo/tree/main/src/solution) contains an implementation that includes all the steps above.

## Remarks

+ All the code that code that has to be changed is within the respective `main.cpp` file in each subdirectory. Most of the code is already prepared. In order to implement the exercises, it is sufficient to insert your code after the blocks
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

   + $`N^2`$ subdomains in two dimensions and
   + $`N^3`$ subdomains in three dimensions,
   
   for some $`N`$.
   
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
     --m                    int           H/h (default 10)
                                          (default: --m=10)
     --1lvl                 bool          Preconditioner type: '1lvl' or '2lvl'
     --2lvl                               (default: --1lvl)
     --xml                  string        File name of the parameter list (default ParameterList.xml).
                                          (default: --xml="ParameterList.xml")
     --epetra               bool          Linear algebra framework: 'epetra' or 'tpetra' (default)
     --tpetra                             (default: --tpetra)
     --v                    int           Verbosity Level. VERB_DEFAULT=-1, VERB_NONE=0 (default), VERB_LOW=1, VERB_MEDIUM=2, VERB_HIGH=3, VERB_EXTREME=4
                                          (default: --v=0)
     --write                bool          Write VTK files of the partitioned solution: 'write' or 'no-write' (default)
     --no-write                           (default: --no-write)
     --timers               bool          Show timer overview: 'timers' or 'no-timers' (default)
     --no-timers                          (default: --no-timers)
   ```

+ Each exercise comes with one or two parameter list files for specifying the settings of the iterative solver and the preconditioner.
