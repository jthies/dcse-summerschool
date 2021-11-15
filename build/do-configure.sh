#!/bin/bash

# Copyright 2021 Alexander Heinlein
# Contact: Alexander Heinlein (a.heinlein@tudelft.nl)

rm -rf CMake*

TRILINOS=/opt/trilinos/install
VTK=/opt/vtk/build

cmake \
  -D Trilinos_PATH:PATH="${TRILINOS}" \
  -D VTK_ENABLE:BOOL=ON \
  -D VTK_DIR:PATH="${VTK}" \
  -D Boost_ENABLE:BOOL=ON \
  -D Boost_LIBRARY_DIR:PATH="/usr/lib64/" \
  -D Boost_INCLUDE_DIR:PATH="/usr/include/" \
  -D CMAKE_BUILD_TYPE:STRING=RELEASE \
  -D BUILD_SHARED_LIBS:BOOL=ON \
  -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
  -D CMAKE_CXX_EXTENSIONS:BOOL=OFF \
  ../src
