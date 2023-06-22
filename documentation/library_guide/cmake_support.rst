CMake Support
#############

General Usage
=============
`CMake <https://cmake.org/cmake/help/latest/index.html>`_ is a cross-platform build system generator.  It can be used to generate build scripts which can then be used to build and link a users application.

``oneDPLConfig.cmake`` and ``oneDPLConfigVersion.cmake`` are distributed with |onedpl_short|.  These files allow to integratration of |onedpl_short| into user projects with the `find_package <https://cmake.org/cmake/help/latest/command/find_package.html>`_ command. Successful invocation of ``find_package(oneDPL <options>)`` creates imported target `oneDPL` that can be passed to the `target_link_libraries <https://cmake.org/cmake/help/latest/command/target_link_libraries.html>`_ command.

Some useful CMake variables (`here <https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html>`_ you can find a full list of CMake variables for the latest version):

- `CMAKE_CXX_COMPILER <https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html>`_ - C++ compiler used for build, e.g. ``CMAKE_CXX_COMPILER=dpcpp``.
- `CMAKE_BUILD_TYPE <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_ - build type that affects optimization level and debug options, values: ``RelWithDebInfo``, ``Debug``, ``Release``, ...; e.g. ``CMAKE_BUILD_TYPE=RelWithDebInfo``.
- `CMAKE_CXX_STANDARD <https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html>`_ - C++ standard, e.g. ``CMAKE_CXX_STANDARD=17``.

|onedpl_short| Backend Selection
==============

The |onedpl_short| backend is selected based on compiler and environment availability, in combination with the user defined ``ONEDPL_PAR_BACKEND`` option. ``ONEDPL_PAR_BACKEND`` is optional, and has three possible valid options: ``tbb`` (oneTBB), ``openmp`` (OpenMP), and ``serial``.  If SYCL is supported by the compiler, ``dpcpp`` backend is always selected.  If this ``ONEDPL_PAR_BACKEND`` is not set then the first suitable backend is chosen among oneTBB, OpenMP and serial, in that order.  |onedpl_short| is considered as not found (``oneDPL_FOUND=FALSE``) if ``ONEDPL_PAR_BACKEND`` is specified, but not found or not supported.

Using |onedpl_short| Package On Windows
===============================
On Windows, some workarounds may be required to use icx[-cl] successfully with |onedpl_short|.  We recommend updating to the most recent version of CMake to minimize the workarounds required for succesful use.  A CMake package has been provided, ``oneDPLWindowsIntelLLVM``, to provide the necessary workarounds to enable support for icx[-cl] on Windows with CMake versions 3.20 and greater.  Some workarounds are provided for icpx, but it is not fully supported on Windows at this time.  To use this package, please add ``find_package(oneDPLWindowsIntelLLVM)`` to your CMake file *before* you call ``project()``.

The supported generator in the Windows environment is `Ninja <https://ninja-build.org/>`_ as described `here <https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/current/use-cmake-with-the-compiler.html>`_.  We recommend using ``-GNinja`` in your CMake configuration to specify your `CMake Generator <https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#ninja-generators>`_.

Example CMake File
==================
To use |onedpl_short| with CMake, a user must create a ``CMakeLists.txt`` file for their project and add |onedpl_short|.  This file should be placed in the project's base directory.  Below is an example ``CMakeLists.txt`` file:

.. code:: cpp

  # only required on Windows
  find_package(oneDPLWindowsIntelLLVM)
  
  project(Foo)
  add_executable(foo foo.cpp)
  
  # Search to find oneDPL
  find_package(oneDPL REQUIRED)
  
  # Connect oneDPL to foo
  target_link_libraries(foo oneDPL)

Example CMake Invocation
========================
After creating a ``CMakeLists.txt`` file for their project, a user may use a command line CMake invocation to generate build scripts for their project.

Below is an example Linux CMake invocation which generates build scripts for the project with the icpx compiler, tbb backend and release build type:

.. code:: cpp

  mkdir build && cd build
  cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=release -DONEDPL_PAR_BACKEND=tbb ..

Below is an example Windows CMake invocation which generates Ninja build scripts for the project in the parent directory with the icx compiler, OpenMP backend and debug build type:

.. code:: cpp

  mkdir build && cd build
  cmake -GNinja -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=debug -DONEDPL_PAR_BACKEND=openmp ..

Both of these examples assume the starting working directory is the project's base directory which contains ``CMakeLists.txt``.  The build scripts are generated in a newly created ``build`` directory.


Example Build command
=====================
Once build scripts have been generated for your desired configuration following the instruction above, a `build command <https://cmake.org/cmake/help/latest/manual/cmake.1.html#build-a-project>`_ can be issued to build your project:

.. code:: cpp

  cmake --build .

This example assumes the starting working directory is in the directory which contains the CMake generated build scripts, ``build``, if following the instructions above.

