# Executable
add_executable(solution-example-2.exe main.cpp)

target_include_directories(solution-example-2.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(solution-example-2.exe
  ${LIBS}
  )

# Copy files

# Tests
