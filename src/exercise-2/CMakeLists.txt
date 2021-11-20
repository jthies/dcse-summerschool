# Executable
add_executable(example-2.exe main.cpp)

target_link_libraries(example-2.exe
  ${LIBS}
  )

# Copy files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/parameters-2d.xml
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/parameters-3d.xml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Tests
