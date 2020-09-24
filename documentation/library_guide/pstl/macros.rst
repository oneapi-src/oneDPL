Macros
#######

Version Macros
===============

================================= ==============================
Macro                             Description
================================= ==============================
``_PSTL_VERSION``                 Current Parallel STL version in LLVM. The value is a decimal numeral of the form ``xxyyz`` where ``xx`` is the major version number, ``yy`` is the minor version number and ``z`` is the patch number.
--------------------------------- ------------------------------
``_PSTL_VERSION_MAJOR``           ``_PSTL_VERSION/1000``; that is, the major version number.
--------------------------------- ------------------------------
``_PSTL_VERSION_MINOR``           ``(_PSTL_VERSION % 1000) / 10``; that is, the minor version number.
--------------------------------- ------------------------------
``_PSTL_VERSION_PATCH``           ``_PSTL_VERSION % 10``; that is, the patch number.
================================= ==============================

Additional Macros
==================

================================= ==============================
Macro                             Description
================================= ==============================
``PSTL_USE_PARALLEL_POLICIES``    This macro controls the use of parallel policies. When set to 0, it disables the ``par``, ``par_unseq`` and DPC++ policies, making their use a compilation error. It's recommended for code that only uses vectorization with ``unseq`` policy, to avoid dependency on the TBB and DPC++ runtime libraries. When the macro is not defined (default) or evaluates to a non-zero value all execution policies are enabled. This macro is a part of Parallel STL code in LLVM.
--------------------------------- ------------------------------
``PSTL_USE_NONTEMPORAL_STORES``   This macro enables the use of ``#pragma vector nontemporal`` in the algorithms ``std::copy``, ``std::copy_n``, ``std::fill``, ``std::fill_n``, ``std::generate``, ``std::generate_n``, ``std::move``, ``std::rotate``, ``std::rotate_copy``, ``std::swap_ranges`` with the ``unseq`` policy. For further details about the pragma, see the `User and Reference Guide for the Intel® C++ Compiler <https://software.intel.com/en-us/node/524559>`_. If the macro evaluates to a non-zero value, the use of ``#pragma vector nontemporal`` is enabled. When the macro is not defined (default) or set to 0, the macro does nothing. This macro is a part of Parallel STL code in LLVM.
--------------------------------- ------------------------------
``PSTL_USAGE_WARNINGS``           This macro enables Parallel STL to emit compile-time messages, such as warnings about an algorithm not supporting a certain execution policy. When set to 1, the macro allows the implementation to emit usage warnings. When the macro is not defined (default) or evaluates to zero, usage warnings are disabled. This macro is a part of Parallel STL code in LLVM.
--------------------------------- ------------------------------
``ONEDPL_STANDARD_POLICIES_ONLY`` This macro disables the use of the DPC++ policies. (This is disabled by default when compiling with the Intel® oneAPI DPC++ Compiler.)
--------------------------------- ------------------------------
``ONEDPL_FPGA_DEVICE``            Use this macro to build your code containing Parallel STL algorithms for FPGA devices. (Disabled by default.)
--------------------------------- ------------------------------
``ONEDPL_FPGA_EMULATOR``          Use this macro to build your code containing Parallel STL algorithms for FPGA emulation device. (Disabled by default.)
================================= ==============================

:Note: Define both ``ONEDPL_FPGA_DEVICE`` and ``ONEDPL_FPGA_EMULATOR`` macros in the same application to run on FPGA emulation device. To run on FPGA hardware device only ``ONEDPL_FPGA_DEVICE`` should be defined.
