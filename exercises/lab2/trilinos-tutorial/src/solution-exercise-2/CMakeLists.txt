# Executable
add_executable(solution-exercise-2.exe main.cpp)

target_include_directories(solution-exercise-2.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(solution-exercise-2.exe
  ${LIBS}
  )

# Copy files

# Tests
