# Copyright 2021 Alexander Heinlein
# Contact: Alexander Heinlein (a.heinlein@tudelft.nl)

# Checking the CMake version
cmake_minimum_required(VERSION 3.17)

project("FROSch Example")

# Looking for Trilinos and printing of the corresponding parameters
if(NOT DEFINED Trilinos_DIR)
  set(Trilinos_DIR "${Trilinos_PATH}/include")
endif()
set(CMAKE_PREFIX_PATH ${Trilinos_PATH} ${CMAKE_PREFIX_PATH})
find_package(Trilinos COMPONENTS REQUIRED ShyLU_DDFROSch)

IF(Trilinos_FOUND)
   MESSAGE("\nFound Trilinos!  Here are the details: ")
   MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
   MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
   MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
   MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
   MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
   MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
   MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
   MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
   MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
   MESSAGE("   Trilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
   MESSAGE("   Trilinos_C_COMPILER = ${Trilinos_C_COMPILER}")
   MESSAGE("   Trilinos_Fortran_COMPILER = ${Trilinos_Fortran_COMPILER}")
   MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
   MESSAGE("   Trilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
   MESSAGE("   Trilinos_Fortran_COMPILER_FLAGS =
     ${Trilinos_Fortran_COMPILER_FLAGS}")
   MESSAGE("   Trilinos_LINKER = ${Trilinos_LINKER}")
   MESSAGE("   Trilinos_EXTRA_LD_FLAGS = ${Trilinos_EXTRA_LD_FLAGS}")
   MESSAGE("   Trilinos_AR = ${Trilinos_AR}")
   MESSAGE("End of Trilinos details\n")
ELSE()
  MESSAGE(FATAL_ERROR "Could not find Trilinos!")
ENDIF()

SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
set(CMAKE_C_FLAGS "${Trilinos_C_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")

include_directories(".")
include_directories(${Trilinos_TPL_INCLUDE_DIRS})
include_directories(${Trilinos_INCLUDE_DIRS})

link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

set(LIBS)

list(APPEND LIBS ${Trilinos_LIBRARIES})
list(APPEND LIBS ${Trilinos_TPL_LIBRARIES})   

# VTK
IF(${VTK_ENABLE})
  find_package(VTK REQUIRED)
  include_directories(${VTK_INCLUDE_DIRS})
  link_directories(${VTK_LIBRARY_DIRS})
  list(APPEND LIBS ${VTK_LIBRARIES}) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D HAVE_VTK")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D HAVE_VTK")
ENDIF()

# Boost
IF(${Boost_ENABLE})
  find_package(Boost COMPONENTS system filesystem REQUIRED)
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
  list(APPEND LIBS ${Boost_LIBRARIES}) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D HAVE_Boost")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D HAVE_Boost")
ENDIF()

list(REMOVE_DUPLICATES DIRS)

enable_testing()