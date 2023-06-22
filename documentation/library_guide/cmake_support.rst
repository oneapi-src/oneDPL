CMake Support
#############

General Usage
=============
oneDPLConfig.cmake and oneDPLConfigVersion.cmake are included into |onedpl_short| distribution.

These files allow to integrate |onedpl_short| into user project with the `find_package <https://cmake.org/cmake/help/latest/command/find_package.html>`_ command. Successful invocation of ``find_package(oneDPL <options>)`` creates imported target `oneDPL` that can be passed to the `target_link_libraries <https://cmake.org/cmake/help/latest/command/target_link_libraries.html`_ command.

Some useful CMake variables (`here https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html>`_ you can find a full list of CMake variables for the latest version):

- `CMAKE_CXX_COMPILER <https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html>`_ - C++ compiler used for build, e.g. ``CMAKE_CXX_COMPILER=dpcpp``.
- `CMAKE_BUILD_TYPE <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_ - build type that affects optimization level and debug options, values: ``RelWithDebInfo``, ``Debug``, ``Release``, ...; e.g. ``CMAKE_BUILD_TYPE=RelWithDebInfo``.
- `CMAKE_CXX_STANDARD <https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html>`_ - C++ standard, e.g. ``CMAKE_CXX_STANDARD=17``.

|onedpl_short| backend
==============

The |onedpl_short| backend is selected based on compiler and environment availability and the user defined ``ONEDPL_PAR_BACKEND`` option.  ``DPCPP`` backend is always selected if SYCL is available. If this ``ONEDPL_PAR_BACKEND`` is not set then the first suitable backend is chosen among oneTBB, OpenMP and serial, in that order.  |onedpl_short| is considered as not found (``oneDPL_FOUND=FALSE``) if ``ONEDPL_PAR_BACKEND`` is specified, but not found or not supported.

Using |onedpl_short| Package On Windows
===============================
On Windows, we recommend updating to the most recent version of CMake, as they are actively `improving support for Intel compilers <https://gitlab.kitware.com/cmake/cmake/-/issues/24314>`_.  CMake requires some workarounds to use icx[-cl] successfully.  A CMake package has been provided 'oneDPLWindowsIntelLLVM' to apply these required workarounds on Windows with CMake versions 3.20+.  Some workarounds are provided for icpx, but it is not fully supported on Windows at this time.  To enable the workarounds, please add ``find_package(oneDPLWindowsIntelLLVM)`` to your CMake file before you call ``project()``.
The supported generator in the Windows environment is Ninja, we recommend using ``-GNinja`` in your cmake configuration.

Example CMake File
==================
To use |onedpl_short| with CMake, create a CMakeLists.txt file for your project and add |onedpl_short|.
For example:

.. code:: cpp

  # only necessary on Windows
  find_package(oneDPLWindowsIntelLLVM)
  
  project(Foo)
  add_executable(foo foo.cpp)
  
  # Search for oneDPL
  find_package(oneDPL REQUIRED)
  
  # Connect oneDPL to foo
  target_link_libraries(foo oneDPL)

Example CMake Invocation
========================
CMake generates build scripts which can then be used to build and link your application.

Below is an example Linux CMake invocation which generates build scripts for the project in the parent directory using tbb backend and release build type: 

.. code:: cpp

  mkdir build && cd build
  cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=release -DONEDPL_PAR_BACKEND=tbb ..

Below is an example Windows CMake invocation which generates Ninja build scripts for the project in the parent directory using OpenMP backend and debug build type: 

.. code:: cpp

  mkdir build && cd build
  cmake -GNinja -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=debug -DONEDPL_PAR_BACKEND=openmp ..
