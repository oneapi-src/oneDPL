# Add an executable target from source files
add_executable(${PROJECT_NAME} main.cpp)

# Specify compilation flags for the target
target_compile_options(${PROJECT_NAME} PRIVATE -fsycl-unnamed-lambda)

if(WIN32)
  # Specify libraries to link with
  target_link_libraries(${PROJECT_NAME} sycl)

  # Add custom target for running
  add_custom_target(run ${PROJECT_NAME}.exe)
else()
  # Add custom target for running
  add_custom_target(run ./${PROJECT_NAME})
endif()