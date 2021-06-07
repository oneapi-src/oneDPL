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
