add_executable(gamma_correction gamma_correction.cpp utils.cpp)
target_link_libraries(gamma_correction PUBLIC pstl::ParallelSTL)
