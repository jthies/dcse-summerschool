# FROSch Demo

This repository contains material for demonstrating, learning, and testing the use of the [FROSch](https://shylu-frosch.github.io) (Fast and Robust Overlapping Schwarz) solver package.

> *FROSch* (*F*ast and *R*obust *O*verlapping *Sch*warz) is a framework for parallel overlapping Schwarz preconditioners in Trilinos. It is designed to be used as an algebraic solver for discrete problems arising from continuum mechanical problems. The framework is part of the Trilinos package ShyLU and it offers a range of functions for the construction and combination of various Schwarz preconditioners.
>
> ---
>
> See [https://shylu-frosch.github.io](https://shylu-frosch.github.io).

In order to make the material

+ easy to use,
+ accessible, and
+ portable,

the repository provides **scripts to build a [Docker](https://www.docker.com) container with all required software**. Within the Docker container, it is easy to compile, run, and modify the **exercises and tutorials for FROSch**.

**Note:** *Of course, the required software can also be installed directly, such that the examples can be compiled and run without the Docker container.*

## Docker container

The Docker container is based on the **latest version of CentOS**. Furthermore, after completing the built, it will contain (among others)

+ **cmake**,
+ **git**,
+ **C, C++, and, Fortran compilers**,
+ **MPICH**,
+ **BLAS**,
+ **LAPACK**, and
+ **Boost**

from the package manager `yum` as well as

+ **SuiteSparse 5.4.0** and
+ **VTK 9.1.0**.

Furthermore, the Docker container will contain **[Trilinos](https://github.com/trilinos/Trilinos/) in the [13.2 release version](https://github.com/trilinos/Trilinos/releases/tag/trilinos-release-13-2-0)**. The Trilinos installation will be a minimum built necessary for the FROSch exercises and tutorials. Among others, this includes the relevant Trilinos packages

+ **Amesos2** *Direct sparse solvers (e.g., Umfpack from Suitesparse)*
+ **Belos** *Iterative solvers*
+ **Galeri** *Example linear systems*
+ **ShyLUDD_FROSch** *FROSch*
+ **Teuchos** *Tools, such as smart pointers, parameter lists, and many more...*
+ **Tpetra** *Parallel linear algebra*
  (FROSch uses Tpetra through the **Xpetra** interface)

In case, you want to modify the configuration of Trilinos (e.g., enable debugging flags or additional packages), please modify the configure script `configurefiles/do-configure-trilinos.sh`. **Be careful when changing any of the paths** as they are chosen according to the configuration of the docker container; similarly, you can modify the configuration of VTK (`configurefiles/do-configure-vtk.sh`).

### Installing Docker

[Docker](https://www.docker.com) can be easily installed on Linux, Windows, and MacOS and facilitates the delivery software in packages called containers, which are isolated and bundle their own software, libraries, and configuration files.

Please first install Docker on your computer following the instructions on the [official website](https://www.docker.com/get-started).

---

**All following steps require that the software Docker is running on your computer. Moreover, if Docker complains about permissions, you should execute the scripts with `sudo`**.

---

### Building the Docker container

The Docker container can be built by executing the script

```shell
./build-container.sh
```

in the main directory of the repository. Running this script will set up a docker image with the flag `frosch_demo` as described above. Since all the above mentioned software will be installed, **this step will take some time**.

In order to **test if the Trilinos and FROSch installation has been successful**, [run the docker container](#running-the-docker-container) (see below) and run the Trilinos tests using

```shell
cd /opt/trilinos/build
ctest
```

If all tests (approx. 150 tests) are successful, the Docker container is ready to be used.

### Running the Docker container

In order to run the Docker container, just execute the script

```shell
./run-container.sh
```

from the main directory of the repository. The script will **run the container** and **mount the current directory** (should be the main directory of the repository) as the local directory `/opt/frosch_demo` within the container.

### Deleting the Docker container

In case you later want to remove the Docker image and the respective mounted volume (from running the container), just execute the script
```shell
./delete-container.sh
```

in the main directory of the repository.

## Exercises & tutorials

The exercises and corresponding explanations can be found in the subdirectory `src` (and the `README.md` files); click [here](https://github.com/searhein/frosch-demo/tree/main/src/). In order to configure the corresponding cmake project and compile the tests, please perform the following steps:


1. In case you are using the Docker container, run the Docker container as described [above](#running-the-docker-container).

2. Enter the `build` directory.

3. If you are using the Docker container, it is sufficient to execute the script
   ```shell
   ./run-container.sh
   ```
   In case, you installed the software requirements manually, you will have to adjust the paths

   ```shell
   TRILINOS=/opt/trilinos/install
   VTK=/opt/vtk/build
   ...
   -D Boost_LIBRARY_DIR:PATH="/usr/lib64/" \
   -D Boost_INCLUDE_DIR:PATH="/usr/include/" \
   ```

   in the script accordingly before executing the script. If you want to compile without VTK and Boost, you can disable `VTK_ENABLE` and `Boost_ENABLE`. This will automatically disable writing the solutions to files.

   **Note:** *You can ignore the warning regarding the `Trilinos_DIR`*.

4. Once the configuration with cmake has finished successfully, you can compile all examples using

   ```shell
   make
   ```

5. To make sure that compilation has been successful and that you are ready to work on the exercises, please run:

   ```shell
   ctest
   ```

   **Note:** *`ctest` will perform a total of 8 tests:*
   + *2D Laplace with a one-level Schwarz preconditioner*
   + *2D Laplace with a GDSW preconditioner*
   + *2D linear elasticity with a one-level Schwarz preconditioner*
   + *2D linear elasticity with a GDSW preconditioner*
   + *3D Laplace with a one-level Schwarz preconditioner*
   + *3D Laplace with a GDSW preconditioner*
   + *3D linear elasticity with a one-level Schwarz preconditioner*
   + *3D linear elasticity with a GDSW preconditioner*

## Additional references

+ **Trilinos**

  + [Website](https://trilinos.github.io/index.html)
  + [GitHub repository](https://github.com/trilinos/Trilinos)
  + [Documentation](https://trilinos.github.io/documentation.html)
  + [Getting started](https://trilinos.github.io/getting_started.html)
  + [Hands-on tutorials](https://github.com/trilinos/Trilinos_tutorial/wiki/TrilinosHandsOnTutorial)

+ **FROSch**

  + [Website](https://shylu-frosch.github.io)

  + References:

    ```
    @article{Heinlein:2016:PIT,
      author = {Heinlein, Alexander and Klawonn, Axel and Rheinbach, Oliver},
      title = {A parallel implementation of a two-level overlapping {S}chwarz method with energy-minimizing coarse space based on {T}rilinos},
      journal = {SIAM J. Sci. Comput.},
      number = {6},
      pages = {C713--C747},
      volume = {38},
      year = {2016},
      doi = {10.1137/16M1062843},
      note = {Preprint \url{http://tu-freiberg.de/sites/default/files/media/fakultaet-fuer-mathematik-und-informatik-fakultaet-1-9277/prep/2016-04_fertig.pdf}}
    }
    
    @inbook{Heinlein:2020:FRO,
      author = {Alexander Heinlein and Axel Klawonn and Sivasankaran Rajamanickam and Oliver Rheinbach},
      title = {{FROSch}: A Fast And Robust Overlapping {S}chwarz Domain Decomposition Preconditioner Based on {X}petra in {T}rilinos},
      booktitle = {Domain Decomposition Methods in Science and Engineering XXV},
      pages = {176--184},
      publisher = {Springer International Publishing},
      year = {2020},
      doi = {10.1007/978-3-030-56750-7_19},
      note = {Preprint \url{https://kups.ub.uni-koeln.de/9018/}}
    }
    ```
