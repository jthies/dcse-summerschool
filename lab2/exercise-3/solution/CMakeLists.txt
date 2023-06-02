project(DCSE_Summerschool_Lab3 CXX)
cmake_minimum_required(VERSION 3.10)

find_package(Trilinos REQUIRED COMPONENTS Teuchos Tpetra Belos Ifpack2)
set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
set(CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS})

add_executable(main.x main.cpp)

target_include_directories(main.x PRIVATE ${Trilinos_INCLUDE_DIRS})
target_link_libraries(main.x ${Trilinos_LIBRARIES})
