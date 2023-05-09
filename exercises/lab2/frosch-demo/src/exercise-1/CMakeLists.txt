# Executable
add_executable(example-1.exe main.cpp)

target_link_libraries(example-1.exe
  ${LIBS}
  )

# Copy files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/parameters.xml
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/parameters.xml
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Tests
