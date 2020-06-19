Before You Begin
#################
Before you begin content for the Intel® oneAPI DPC++ Library (oneDPL).

Install the `Intel® oneAPI Base Toolkit <https://software.intel.com/en-us/oneapi/base-kit>`_ to use oneDPL.

To use Parallel STL or the Extension API, include the corresponding header files in your source code. All oneDPL header files are in the ``dpstd`` directory. Use ``#include <dpstd/…>`` to include them. Intel® oneAPI DPC++ Library uses namespace ``dpstd`` for the Extension API classes and functions.

To use tested C++ standard APIs, you need to include the corresponding C++ standard header files and use the ``std`` namespace.
