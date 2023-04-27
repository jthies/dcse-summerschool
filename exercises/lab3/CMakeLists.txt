project(DCSE_Summerschool_Lab3 CXX)
cmake_minimum_required(VERSION 3.10)

find_package(Trilinos REQUIRED COMPONENTS Teuchos Belos Ifpack2 Tpetra)
set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})

add_executable(ifpack_driver.x ifpack_driver.cpp)

message(STATUS "${Trilinos_INCLUDE_DIRS}")

target_include_directories(ifpack_driver.x PRIVATE ${Trilinos_INCLUDE_DIRS})
target_link_libraries(ifpack_driver.x ${Trilinos_LIBRARIES})
