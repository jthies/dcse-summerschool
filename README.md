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

Furthermore, the Docker container will contain **[Trilinos](https://github.com/trilinos/Trilinos/) in the [13.2 release version](https://github.com/trilinos/Trilinos/releases/tag/trilinos-release-13-2-0)**. The Trilinos installation will be a minimum built necessary for the FROSch exercises and tutorials. In case, you want to modify the configuration of Trilinos (e.g., enable debugging flags or additional packages), please modify the configure script `configurefiles/do-configure-trilinos.sh`. **Be careful when changing any of the paths** as they are chosen according to the configuration of the docker container; similarly, you can modify the configuration of VTK (`configurefiles/do-configure-vtk.sh`).

### Installing Docker

[Docker](https://www.docker.com) can be easily installed on Linux, Windows, and MacOS and facilitates the delivery software in packages called containers, which are isolated and bundle their own software, libraries, and configuration files. 

Please first install Docker on your computer following from the [official website](https://www.docker.com/get-started). 

---

**All following steps require that the software Docker is running on your computer**

---

### Building the Docker container 

The Docker container can be built by executing the script

```shell
./build-container.sh
```

in the main directory of the repository. Running this script will set up a docker image with the flag `frosch_demo` as described above. Since all the above mentioned software will be installed, this step will take some time. 

In order to **test if the Trilinos and FROSch installation has been successful**, [run the docker container](#running-the-docker-container) (see below) and run the Trilinos tests using

```shell
cd /opt/trilinos/build
ctest
```

If all tests are successful, the Docker container is ready to be used.

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