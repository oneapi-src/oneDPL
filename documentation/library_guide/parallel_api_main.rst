Parallel API
############

Introduction to Parallel API
============================

Algorithms
==========

Parallel STL is an implementation of the C++ standard library algorithms with support for execution
policies, as specified in ISO/IEC 14882:2017 standard, commonly called C++17. The implementation also
supports the unsequenced execution policy and the algorithms shift_left and shift_right, which are specified
in the final draft for the C++ 20 standard (N4860). For more details see the `C++ Reference Standard Execution
Policies <https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t>`_.

Parallel STL offers efficient support for both parallel and vectorized execution of
algorithms for Intel® processors. For sequential execution, it relies on an available
implementation of the C++ standard library. 

The Extension API currently includes algorithms, iterators, and function object classes. The algorithms
include segmented reduce, segmented scan and vectorized search algorithms. The iterators provided implement
zip, transform, and permutation operations on other iterators, and also include a counting iterator
and a discard iterator. The function object classes provide minimum, maximum and identity operations
that may be passed to algorithms such as reduce or transform. The Extension API also includes an experimental
implementation of range-based algorithms and the ranges required to use them.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :glob:

   parallel_api/execution_policies
   parallel_api/algorithms_advanced_use_cases
   parallel_api/iterators_advanced_use_cases
   parallel_api/async_api
   parallel_api/range_based_api
   parallel_api/buffers_and_usm
