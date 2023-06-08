# Lab 4: Performance optimization of communication-avoiding algorithms

In this lab, we will construct and test various Krylov variants with reduced impact of orthogonalization
("communication-avoiding" methods. As test problems, we will use the matrices you downloaded for yesterday's ``lab3``
(e.g. the 'Transport' matrix). Two complete linear systems including right-hand side (rhs) and solution (sln) can 
be found in ``/beegfs/apps/unsupported/summerschool/. For quick checks, use the small matrix lap128x128.mm, because the MomentumEQS matrix 
(which comes from a wind-turbine simulation) takes about a minute to read from the file (on DelftBlue).

## What you will practice here

- working with Trilinos Belos/Tpetra/Ifpack2
- tuning solver parameters
- performance analysis using likwid

## Compiling and running the executable

As before, use the following series of commands to build the application:
```bash
source env.sh
mkdir build
cd build
cmake ..
make
```

Note:
- The following command gives you a list of all command-line options supported by the driver routine (main program):
```bash
./ifpack_driver.x --help |& less
```
- The flag ``--listMethods`` prints some valid optins for the ``--solverTypes`` and ``--preconditionerTypes`` arguments.
- In order to reduce the overhead of reading the large ASCII matrix file, you can run multiple solver/preconditioner
cobinations by passing in a comma-separated list, e.g.: ``preconTypes="None,RELAXATION,ILUT". _Do not add whitespace after the commas_.
- A sample job script is included.

## Your tasks

- Try some different preconditioners in combination with the GMRES solver to investigate what gives fast convergence, fast solve time,
and fast overall time (including construction of the preconditioner).
- Then switch to one of the matrices from yesterday's exercise. Look into the header of the matrix file to
get some hints, and try different solver/preconditioner combinations.
- Try running your optimal solver choices on a GPU. Are they still the best choice?

## Use Profiling to guide your parameter-tuning

The choice of solver and preconditioner may shift the balance between the different types of
basic operations (e.g. SpMV vs. BLAS1 vector operations). You can observe this effect in the 
detailed timing output printed after each solve.
