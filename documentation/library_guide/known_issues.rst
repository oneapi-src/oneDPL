.. _known_issues:

Known Issues and Limitations
############################

Existing Issues
^^^^^^^^^^^^^^^

- The definition of lambda functions used with parallel algorithms should not depend on preprocessor macros
  that makes it different for the host and the device. Otherwise, the behavior is undefined.
- ``exclusive_scan`` and ``transform_exclusive_scan`` algorithms may provide wrong results with
  vector execution policies when building a program with GCC 10 and using -O0 option.
- The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to
  compilation errors (caused by oneTBB API changes). 
  To overcome these issues, include oneDPL header files before the standard C++ header files,
  or disable parallel algorithms support in the standard library. 
  For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  ``oneapi::dpl`` namespace, or create a namespace alias. 
- The use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher)
  or Clang 7 (or higher).
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
- Due to specifics of Microsoft* Visual C++, some standard floating-point math functions
  (including ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision. 

.. _`Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`: https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-threading-building-blocks-release-notes.html