project(DCSE_Summerschool_Lab4 CXX)
cmake_minimum_required(VERSION 3.10)

find_package(Trilinos REQUIRED COMPONENTS Teuchos Belos Ifpack2 Tpetra)
set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
set(CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS})

add_executable(ifpack_driver.x ifpack_driver.cpp)

target_include_directories(ifpack_driver.x PRIVATE ${Trilinos_INCLUDE_DIRS})
target_link_libraries(ifpack_driver.x ${Trilinos_LIBRARIES})

find_package(likwid REQUIRED CONFIG)
target_link_libraries(ifpack_driver.x ${LIKWID_LIBRARY})
target_include_directories(ifpack_driver.x PRIVATE ${LIKWID_INCLUDE_DIR})
add_definitions(-DLIKWID_PERFMON)
