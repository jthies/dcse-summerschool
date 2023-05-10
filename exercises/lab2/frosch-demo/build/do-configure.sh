#!/bin/bash

# Copyright 2021 Alexander Heinlein
# Contact: Alexander Heinlein (a.heinlein@tudelft.nl)

rm -rf CMake*

TRILINOS=/opt/trilinos/install
VTK=/opt/vtk/build

cmake \
  -D Trilinos_PATH:PATH="${TRILINOS}" \
  -D VTK_ENABLE:BOOL=OFF \
  -D Boost_ENABLE:BOOL=OFF \
  -D CMAKE_BUILD_TYPE:STRING=RELEASE \
  -D BUILD_SHARED_LIBS:BOOL=ON \
  -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
  -D CMAKE_CXX_EXTENSIONS:BOOL=OFF \
  ../src
