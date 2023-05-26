# Executable
add_executable(example-2.exe main.cpp)

target_include_directories(example-2.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(example-2.exe
  ${LIBS}
  )

# Copy files

# Tests
