# Executable
add_executable(solution.exe main.cpp)

target_link_libraries(solution.exe
  ${LIBS}
  )

# Copy files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/parameters-2d.xml
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/parameters-3d.xml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Tests
add_test(NAME 2d-laplace-1lvl
  COMMAND mpirun -n 4 ${CMAKE_CURRENT_BINARY_DIR}/solution.exe --dim=2 --eq=laplace --m=100 --prec=1lvl --xml=parameters-2d.xml)

add_test(NAME 2d-laplace-2lvl
  COMMAND mpirun -n 4 ${CMAKE_CURRENT_BINARY_DIR}/solution.exe --dim=2 --eq=laplace --m=100 --prec=2lvl --xml=parameters-2d.xml)

add_test(NAME 2d-elas-1lvl
  COMMAND mpirun -n 4 ${CMAKE_CURRENT_BINARY_DIR}/solution.exe --dim=2 --eq=elas --m=100 --prec=1lvl --xml=parameters-2d.xml)

add_test(NAME 2d-elas-2lvl
  COMMAND mpirun -n 4 ${CMAKE_CURRENT_BINARY_DIR}/solution.exe --dim=2 --eq=elas --m=100 --prec=2lvl --xml=parameters-2d.xml)

add_test(NAME 3d-laplace-1lvl
  COMMAND mpirun -n 8 ${CMAKE_CURRENT_BINARY_DIR}/solution.exe --dim=3 --eq=laplace --m=10 --prec=1lvl --xml=parameters-3d.xml)

add_test(NAME 3d-laplace-2lvl
  COMMAND mpirun -n 8 ${CMAKE_CURRENT_BINARY_DIR}/solution.exe --dim=3 --eq=laplace --m=10 --prec=2lvl --xml=parameters-3d.xml)

add_test(NAME 3d-elas-1lvl
  COMMAND mpirun -n 8 ${CMAKE_CURRENT_BINARY_DIR}/solution.exe --dim=3 --eq=elas --m=10 --prec=1lvl --xml=parameters-3d.xml)

add_test(NAME 3d-elas-2lvl
  COMMAND mpirun -n 8 ${CMAKE_CURRENT_BINARY_DIR}/solution.exe --dim=3 --eq=elas --m=10 --prec=2lvl --xml=parameters-3d.xml)
