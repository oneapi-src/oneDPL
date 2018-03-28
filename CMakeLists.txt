cmake_minimum_required(VERSION 3.1)

# Extract version
set(PARALLELSTL_VERSION_FILE "include/pstl/internal/pstl_config.h")
file(STRINGS "${PARALLELSTL_VERSION_FILE}" PARALLELSTL_VERSION_SOURCE REGEX "#define PSTL_VERSION .*$")
string(REGEX MATCH "#define PSTL_VERSION (.*)$" PARALLELSTL_VERSION_SOURCE "${PARALLELSTL_VERSION_SOURCE}")
math(EXPR VERSION_MAJOR "${PARALLELSTL_VERSION_SOURCE} / 100")
math(EXPR VERSION_MINOR "${PARALLELSTL_VERSION_SOURCE} % 100")

project(ParallelSTL VERSION ${VERSION_MAJOR}.${VERSION_MINOR} LANGUAGES CXX)

set(PARALLELSTL_BACKEND "tbb" CACHE STRING "Threading backend; defaults to TBB")

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 11)
    option(PARALLELSTL_TESTS "Build tests" ON)
    option(PARALLELSTL_EXAMPLES "Build examples" ON)
else()
    option(PARALLELSTL_TESTS "Build tests" OFF)
    option(PARALLELSTL_EXAMPLES "Build examples" OFF)
endif()

# Adds the config version helper
include(CMakePackageConfigHelpers)

add_library(ParallelSTL INTERFACE)

# Support adding as subdirectory with same syntax as find_package targets
add_library(pstl::ParallelSTL ALIAS ParallelSTL)

if(PARALLELSTL_BACKEND STREQUAL "tbb")
    find_package(TBB REQUIRED)
    target_link_libraries(ParallelSTL INTERFACE tbb)
else()
    target_add_definitions(ParallelSTL INTERFACE PSTL_USE_PARALLEL_POLICIES=0)
endif()

target_include_directories(ParallelSTL
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include>)

if(PARALLELSTL_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(PARALLELSTL_EXAMPLES)
    add_subdirectory(examples)
endif()

install(
    TARGETS ParallelSTL
    EXPORT ParallelSTLTargets
    INCLUDES DESTINATION include
    )

install(
    EXPORT ParallelSTLTargets
    FILE ParallelSTLTargets.cmake
    NAMESPACE pstl::
    DESTINATION lib/cmake/ParallelSTL
    )

install(DIRECTORY include DESTINATION include)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelSTLConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

configure_file(
    ParallelSTLConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelSTLConfig.cmake
    @ONLY)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelSTLConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelSTLConfigVersion.cmake
    DESTINATION
    lib/cmake/ParallelSTL
    )

# Allow build directory to work for CMake import
export(TARGETS ParallelSTL NAMESPACE pstl:: FILE ParallelSTLTargets.cmake)
export(PACKAGE ParallelSTL)

set(CPACK_PACKAGE_VENDOR "intel")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Implementation of ISO C++ N4659")
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_SOURCE_DIR}/LICENSE)
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
include(CPack)
