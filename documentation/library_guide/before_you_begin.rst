Before You Begin
#################

Visit the `Intel® oneAPI DPC++ Library Release Notes
<https://software.intel.com/en-us/articles/intel-oneapi-dpcpp-library-release-notes-beta>`_
page for:

- Where to Find the Release
- Overview
- New in this Release
- Known Issues

Install the `Intel® oneAPI Base Toolkit <https://software.intel.com/en-us/oneapi/base-kit>`_
to use the Intel® oneAPI DPC++ Library (oneDPL).

To use Parallel STL or the Extension API, include the corresponding header files in your source code.
All oneDPL header files are in the ``oneapi/dpl`` directory. Use ``#include <oneapi/dpl/…>`` to include them.
Intel® oneAPI DPC++ Library uses namespace ``oneapi::dpl`` for most its classes and functions.

To use tested C++ standard APIs, you need to include the corresponding C++ standard header files
and use the ``std`` namespace.
