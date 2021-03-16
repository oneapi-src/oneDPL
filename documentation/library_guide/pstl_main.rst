Parallel STL Overview
######################

Introduction to Parallel STL
=============================

Parallel STL is an implementation of the C++ standard library algorithms with support for execution
policies, as specified in ISO/IEC 14882:2017 standard, commonly called C++17. The implementation also
supports the unsequenced execution policy and the algorithms shift_left and shift_right, which are specified
in the the final draft for the C++ 20 standard (N4860). For more details see `the standard execution
policies <https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t>`_.

Parallel STL offers efficient support for both parallel and vectorized execution of
algorithms for Intel® processors. For sequential execution, it relies on an available
implementation of the C++ standard library. 

The implementation supports the DPC++ execution policies used to run the massive parallel
computational model for heterogeneous systems. The policies are specified in
`the oneDPL Spec <https://spec.oneapi.com/versions/latest/elements/oneDPL/source/index.html#dpc-execution-policy>`_.

For any of the implemented algorithms, pass one of the execution policies values as the first
argument in a call to the algorithm to specify the desired execution policy. The policies have
the following meaning:

================================= ==============================
Execution policy value            Description
================================= ==============================
``seq``                           Sequential execution.
--------------------------------- ------------------------------
``unseq``                         Unsequenced SIMD execution. This policy requires that
                                  all functions provided are SIMD-safe.
--------------------------------- ------------------------------
``par``                           Parallel execution by multiple threads.
--------------------------------- ------------------------------
``par_unseq``                     Combined effect of ``unseq`` and ``par``.
--------------------------------- ------------------------------
``dpcpp_default``                 Massive parallel execution on devices using DPC++.
--------------------------------- ------------------------------
``dpcpp_fpga``                    Massive parallel execution on FPGA devices.
================================= ==============================

Implementation is based on Parallel STL from
`LLVM <https://github.com/llvm/llvm-project/tree/master/pstl>`_.

Prerequisites
==============

C++11 is the minimal version of the C++ standard, which oneDPL requires. That means, any use of oneDPL
requires at least a C++11 compiler. Some uses of the library may require a higher version of C++.
To use Parallel STL with the C++ standard policies, you must have the following software installed:

  * C++ compiler with support for OpenMP* 4.0 (or higher) SIMD constructs
  * oneAPI Threading Building Blocks (oneTBB) or Threading Building Blocks (TBB) 2019 and later

To use Parallel STL with the DPC++ execution policies, you must have the following software installed:

  * C++ compiler with support for SYCL 2020

.. _pstl:

.. toctree::
   :maxdepth: 1

   pstl/dpcpp_policies_usage
   pstl/macros
