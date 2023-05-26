project(DCSE_Summerschool_Lab3 CXX)
cmake_minimum_required(VERSION 3.10)

find_package(Trilinos REQUIRED COMPONENTS Teuchos Belos Tpetra)
set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
set(CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS})

add_executable(tpetra_driver.x tpetra_driver.cpp)

target_include_directories(tpetra_driver.x PRIVATE ${Trilinos_INCLUDE_DIRS})
target_link_libraries(tpetra_driver.x ${Trilinos_LIBRARIES})
