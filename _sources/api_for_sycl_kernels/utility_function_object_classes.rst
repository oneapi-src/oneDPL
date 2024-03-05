Utility Function Object Classes
##################################

The definitions of the utility function objects are available through the
``oneapi/dpl/functional`` header.  All function objects are implemented in the ``oneapi::dpl`` namespace.

* ``identity``: A C++11 implementation of the C++20 ``std::identity`` function object type, where the operator() returns the
  argument unchanged.
* ``minimum``: A function object type where the operator() applies ``std::less`` to its arguments, then returns the
  lesser argument unchanged.
* ``maximum``: A function object type where the operator() applies ``std::greater`` to its arguments, then returns the
  greater argument unchanged.