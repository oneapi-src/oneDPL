Extension API
#############
The Extension API currently includes algorithms, iterators, and function object classes. The algorithms
include segmented reduce, segmented scan and vectorized search algorithms. The iterators provided implement
zip, transform, and permutation operations on other iterators, and also include a counting iterator
and a discard iterator. The function object classes provide minimum, maximum and identity operations
that may be passed to algorithms such as reduce or transform. The Extension API also includes an experimental
implementation of range-based algorithms and the ranges required to use them.

Parallel Algorithms
-------------------

The definitions of the algorithms provided in the Extension API are available through the ``oneapi/dpl/algorithm``
header.  All algorithms are implemented in the ``oneapi::dpl`` namespace.

* ``reduce_by_segment``

  The ``reduce_by_segment`` algorithm performs partial reductions on a sequence's values and keys. Each
  reduction is computed with a given reduction operation for a contiguous subsequence of values, which are
  determined by keys being equal according to a predicate. A return value is a pair of iterators holding
  the end of the output sequences for keys and values.

  For correct computation, the reduction operation should be associative. If no operation is specified,
  the default operation for the reduction is ``std::plus``, and the default predicate is ``std::equal_to``.
  The algorithm requires that the type of the elements used for values be default constructible. For example::

    keys:   [0,0,0,1,1,1]
    values: [1,2,3,4,5,6]
    output_keys:   [0,1]
    output_values: [1+2+3=6,4+5+6=15]

* ``inclusive_scan_by_segment``

  The ``inclusive_scan_by_segment`` algorithm performs partial prefix scans on a sequence's values. Each
  scan applies to a contiguous subsequence of values, which are determined by the keys associated with the
  values being equal. The return value is an iterator targeting the end of the result sequence.

  For correct computation, the prefix scan operation should be associative. If no operation is specified,
  the default operation is ``std::plus``, and the default predicate is ``std::equal_to``. The algorithm
  requires that the type of the elements used for values be default constructible. For example::

    keys:   [0,0,0,1,1,1]
    values: [1,2,3,4,5,6]
    result: [1,1+2=3,1+2+3=6,4,4+5=9,4+5+6=15]

* ``exclusive_scan_by_segment``

  The ``exclusive_scan_by_segment`` algorithm performs partial prefix scans on a sequence's values. Each
  scan applies to a contiguous subsequence of values that are determined by the keys associated with the values
  being equal, and sets the first element to the initial value provided. The return value is an iterator
  targeting the end of the result sequence.

  For correct computation, the prefix scan operation should be associative. If no operation is specified,
  the default operation is ``std::plus``, and the default predicate is ``std::equal_to``. For example::

    keys:   [0,0,0,1,1,1]
    values: [1,2,3,4,5,6]
    initial value: [0]
    result: [0,0+1=1,0+1+2=3,0,0+4=4,0+4+5=9]

* ``binary_search``

  The ``binary_search`` algorithm performs a binary search of the input sequence for each of the values in
  the search sequence provided.  For each element of the search sequence the algorithm writes a boolean value
  to the result sequence that indicates whether the search value was found in the input sequence. An iterator
  to one past the last value in the result sequence is returned. The algorithm assumes the input sequence has
  been sorted by the comparator provided. If no comparator is provided then a function object that uses
  ``operator<`` to compare the elements will be used. For example::

    input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
    search sequence: [0, 2, 4, 7, 6]
    result sequence: [true, true, false, false, true]

* ``lower_bound``

  The ``lower_bound`` algorithm performs a binary search of the input sequence for each of the values in
  the search sequence provided to identify the lowest index in the input sequence where the search value could
  be inserted without violating the sorted ordering of the input sequence.  The lowest index for each search
  value is written to the result sequence, and the algorithm returns an iterator to one past the last value
  written to the result sequence. If no comparator is provided then a function object that uses ``operator<``
  to compare the elements will be used. For example::

    input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
    search sequence: [0, 2, 4, 7, 6]
    result sequence: [0, 1, 8, 10, 8]

* ``upper_bound``

  The ``upper_bound`` algorithm performs a binary search of the input sequence for each of the values in
  the search sequence provided to identify the highest index in the input sequence where the search value could
  be inserted without violating the sorted ordering of the input sequence.  The highest index for each search
  value is written to the result sequence, and the algorithm returns an iterator to one past the last value
  written to the result sequence. If no comparator is provided then a function object that uses ``operator<``
  to compare the elements will be used. For example::

    input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
    search sequence: [0, 2, 4, 7, 6]
    result sequence: [1, 4, 8, 10, 10]

Iterators
---------

The definitions of the iterators provided in the Extension API are available through the ``oneapi/dpl/iterator``
header.  All iterators are implemented in the ``oneapi::dpl`` namespace.

* ``counting_iterator``

  The ``counting_iterator`` is a random access iterator-like type whose dereferenced value is an integer
  counter. Instances of ``counting_iterator`` provide read-only dereference operations. The counter of an
  ``counting_iterator`` instance changes according to the arithmetic of the random access iterator type::

    using namespace oneapi;
    dpl::counting_iterator<int> count_a(0);
    dpl::counting_iterator<int> count_b = count_a + 10;
    int init = count_a[0]; // OK: init == 0
    *count_b = 7; // ERROR: counting_iterator doesn't provide write operations
    auto sum = std::reduce(dpl::execution::dpcpp_default,
                           count_a, count_b, init); // sum is (0 + 0 + 1 + ... + 9) = 45

* ``discard_iterator``

  The ``discard_iterator`` is a random access iterator-like type that provides write-only dereference
  operations that discard values passed.

  The iterator is useful in the implementation of stencil algorithms where the stencil is not part of the
  desired output. An example of this would be a ``copy_if`` algorithm that receives an an input iterator range
  and a stencil iterator range and copies the elements of the input whose corresponding stencil value is 1. We
  do not want to declare a temporary allocation to store the copy of the stencil, and thus use ``discard_iterator``::

    using namespace oneapi;
    auto zipped_first = dpl::make_zip_iterator(first, stencil);
    std::copy_if(dpl::execution::dpcpp_default,
                 zipped_first, zipped_first + (last - first),
                 dpl::make_zip_iterator(result, dpl::discard_iterator()),
                 [](auto t){return get<1>(t) == 1;}

* ``transform_iterator``

  The ``transform_iterator`` is an iterator defined over another iterator whose dereferenced value is the result
  of a function applied to the corresponding element of the original iterator.  Both the type of the original
  iterator and the unary function applied during dereference operations are required template parameters of
  the ``transform_iterator`` class. The ``transform_iterator`` class provides three constructors:

  * ``transform_iterator()`` instantiates the iterator using default constructed base iterator and unary functor.

  * ``transform_iterator(iter)`` instantiates the iterator using the base iterator provided and a default constructed unary functor.

  * ``transform_iterator(iter, func)`` instantiates the iterator using the base iterator and unary functor provided.
  

  To simplify the construction of the iterator ``oneapi::dpl::make_transform_iterator`` is provided. The
  function receives the original iterator and transform operation instance as arguments, and constructs the
  ``transform_iterator`` instance::

    using namespace oneapi;
    dpl::counting_iterator<int> first(0);
    dpl::counting_iterator<int> last(10);
    auto transform_first = dpl::make_transform_iterator(first, std::negate<int>());
    auto transform_last = transform_first + (last - first);
    auto sum = std::reduce(dpl::execution::dpcpp_default,
                           transform_first, transform_last); // sum is (0 + -1 + ... + -9) = -45

* ``permutation_iterator``

  The ``permutation_iterator`` is an iterator whose dereferenced value set is defined by the source iterator
  provided, and whose iteration order over the dereferenced value set is defined by either another iterator or
  a functor whose index operator defines the mapping from the ``permutation_iterator`` index to the index of the
  source iterator. The ``permutation_iterator`` is useful in implementing applications where noncontiguous
  elements of data represented by an iterator need to be processed by an algorithm as though they were contiguous.
  An example is copying every other element to an output iterator.

  ``make_permutation_iterator`` is provided to simplify construction of iterator instances.  The function
  receives the source iterator and the iterator or function object representing the index map::

    struct multiply_index_by_two {
        template <typename Index>
        Index operator()(const Index& i)
        { return i*2; }
    };

    // first and last are iterators that define a contiguous range of input elements
    // compute the number of elements in the range between the first and last that are accessed
    // by the permutation iterator
    size_t num_elements = std::distance(first, last) / 2 + std::distance(first, last) % 2;
    using namespace oneapi;
    auto permutation_first = dpl::make_permutation_iterator(first, multiply_index_by_two());
    auto permutation_last = permutation_first + num_elements;
    std::copy(dpl::execution::dpcpp_default, permutation_first, permutation_last, result);

* ``zip_iterator``

  The ``zip_iterator`` is an iterator constructed with one or more iterators as input. The value returned by the
  iterator when dereferenced is a tuple of the values returned by dereferencing the member iterators on which
  the ``zip_iterator`` is defined. Arithmetic operations performed on a ``zip_iterator`` instance are also
  applied to each of the member iterators.

  The ``make_zip_iterator`` function is provided to simplify the construction of ``zip_iterator`` instances.
  The function receives each of the iterators to be held as member iterators by the ``zip_iterator`` instance
  it returns.

  The example provided for ``discard_iterator`` demonstrates ``zip_iterator`` use in defining stencil
  algorithms. The ``zip_iterator`` is also useful in defining by key algorithms where input iterators
  representing keys and values are processed as key-value pairs. The example below demonstrates a stable sort
  by key where only the keys are compared but both keys and values are swapped::

    using namespace oneapi;
    auto zipped_begin = dpl::make_zip_iterator(keys_begin, vals_begin);
    std::stable_sort(dpl::execution::dpcpp_default, zipped_begin, zipped_begin + n,
        [](auto lhs, auto rhs) { return get<0>(lhs) < get<0>(rhs); });


Function Object Classes
-----------------------

The definitions of the function objects provided in the Extension API are available through the
``oneapi/dpl/functional`` header.  All function objects are implemented in the ``oneapi::dpl`` namespace.

* ``identity``: A C++11 implementation of the C++20 ``std::identity`` function object type, where the operator() returns the
  argument unchanged.
* ``minimum``: A function object type where the operator() applies ``std::less`` to its arguments, then returns the
  lesser argument unchanged.
* ``maximum``: A function object type where the operator() applies ``std::greater`` to its arguments, then returns the
  greater argument unchanged.

Range-based API
---------------

C++20 indroduces the Ranges library. C++20 standard splits ranges into two categories: factories and adaptors.
A range factory doesn't have underlying data. An element is generated on success by an index or by dereferencing an iterator.
A range adaptor, from the |onedpl_long| perspective, is an utility that transforms base range, or another adapted range into 
a view with custom behavior.

|onedpl_short| supports ``iota_view`` range factory.

``sycl::buffer`` wrapped with ``all_view`` can be used as the range.

The |onedpl_short| considers the supported factories and ``all_view`` as base ranges.
The range adaptors may be combined into a pipeline with a ``base`` range at the beginning. For example::

    cl::sycl::buffer<int> buf(data, cl::sycl::range<1>(10));
    auto range_1 = iota_view(0, 10) | views::reverse();
    auto range_2 = all_view(buf) | views::reverse();

For the range, based on the ``all_view`` factory, data access is permitted on a device only. ``size()`` and ``empty()`` methods are allowed 
to be called on both host and device.

The following algorithms are available to use with the ranges:

* ``all_of``
* ``any_of``
* ``copy``
* ``count``
* ``count_if``
* ``equal``
* ``exclusive_scan``
* ``find``
* ``find_if``
* ``find_if_not``
* ``find_end``
* ``find_first_of``
* ``for_each``
* ``inclusive_scan``
* ``is_sorted``
* ``is_sorted_until``
* ``min_element``
* ``max_element``
* ``merge``
* ``minmax_element``
* ``move``
* ``reduce``
* ``remove``
* ``remove_if``
* ``replace``
* ``replace_if``
* ``search``
* ``sort``
* ``stable_sort``
* ``transform``
* ``transform_reduce``
* ``transform_exclusive_scan``
* ``transform_inclusive_scan``

The signature example of the range-based algorithms looks like::

  template <typename ExecutionPolicy, typename Range1, typename Range2>
  void copy(ExecutionPolicy&& exec, Range1&& source, Range2&& destination);

where ``source`` is used instead of two iterators to represent the input and ``destination`` represents the output.

These algorithms are declared in ``oneapi::dpl::experimental::ranges`` namespace and implemented only for |dpcpp_long| policies.
In order to make these algorithm available the ``<oneapi/dpl/ranges>`` header should be included (after ``<oneapi/dpl/execution>``).
Use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher) or Clang 7 (or higher).

The following views are declared in the ``oneapi::dpl::experimental::ranges`` namespace. Only those are allowed to use as ranges
for range-based algorithms.

* ``all_view`` is a custom utility that represents a view of all or a part of ``sycl::buffer`` underlying elements.
* ``guard_view`` is a custom utility that represents a view of USM data range defined by a two USM pointers.
* ``iota_view`` is a range factory that generates a sequence of N elements, which starts from an initial value and ends by final N-1.
* ``generate`` is a range factory that generates a sequence of N elements, where each is produced by a given functional genrator.
* ``fill`` is a range factory that generates a sequence of N elements, where each is equal a given value.
* ``zip_view`` is a custom range adapter that produces one ``zip_view`` from other several views.
* ``transform_view`` is a range adapter that represents a view of a underlying sequence after applying a transformation to each element.
* ``reverse_view`` is a range adapter that produces a reversed sequence of elements provided by another view.
* ``take_view`` is a range adapter that produces a view of the first N elements from another view.
* ``drop_view`` is a range adapter that produces a view excluding the first N elements from another view.
* ``rotate``: is a range adapter that produces a left rotated sequence of elements provided by another view.

Example of Range-based API Usage
--------------------------------

::

    using namespace oneapi::dpl::experimental::ranges;

    {
        cl::sycl::buffer<int> A(data, cl::sycl::range<1>(max_n));
        cl::sycl::buffer<int> B(data2, cl::sycl::range<1>(max_n));

        auto view = all_view(A) | views::reverse();
        auto range_res = all_view<int, cl::sycl::access::mode::write>(B);

        copy(oneapi::dpl::execution::dpcpp_default, view, range_res);
    }

Async API
--------------------------

The functions defined in the STL ``<algorithm>`` or ``<numeric>`` headers are traditionally blocking. |onedpl_short| extends the
functionality of C++17 parallel algorithms by providing asynchronous algorithms with non-blocking behavior.
This experimental feature enables you to express a concurrent control flow by building dependency chains and interleaving algorithm calls
and interoperability with |dpcpp_short| and SYCL* kernels. 

The current implementation for async algorithms is limited to |dpcpp_short| Execution Policies.
All the functionality described below is available in the ``oneapi::dpl::experimental`` namespace.

The following async algorithms are currently supported:

* ``copy_async``
* ``fill_async``
* ``for_each_async``
* ``reduce_async``
* ``transform_async``
* ``transform_reduce_async``
* ``sort_async``

All the interfaces listed above are a subset of C++17 STL algorithms,
where the suffix ``_async`` is added to the corresponding name (for example: ``reduce``, ``sort``, etc.).
The behavior and signatures are overlapping with the C++17 STL algorithm with the following changes:

* Do not block the execution.
* Take an arbitrary number of events (including 0) as last arguments to allow expressing input dependencies.
* Return future-like object that allows ``wait`` for completion and ``get`` the result.

The type of the future-like object returned from asynchronous algorithm is unspecified. The following member functions are present:

* ``get()`` returns the result.
* ``wait()`` waits for the result to become available.

If the returned object is the result of an algorithm with device policy, it can be converted into a ``sycl::event``.
Lifetime of any resources the algorithm allocates (for example: temporary storage) is bound to the lifetime of the returned object.

Utility functions:

* ``wait_for_all(…)`` waits for an arbitrary number of objects that are convertible into ``sycl::event`` to become ready.


Example of Async API Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

    #include <oneapi/dpl/execution>
    #include <oneapi/dpl/async>
    #include <CL/sycl.hpp>
    
    int main() {
        using namespace oneapi;
        {
            /* Build and compute a simple dependency chain: Fill buffer -> Transform -> Reduce */
            sycl::buffer<int> a{10};
 
            auto fut1 = dpl::experimental::fill_async(dpl::execution::dpcpp_default, 
                                                      dpl::begin(a),dpl::end(a),7);
            
            auto fut2 = dpl::experimental::transform_async(dpl::execution::dpcpp_default,
                                                           dpl::begin(a),dpl::end(a),dpl::begin(a),
                                                           [&](const int& x){return x + 1; },fut1);
            auto ret_val = dpl::experimental::reduce_async(dpl::execution::dpcpp_default,
                                                           dpl::begin(a),dpl::end(a),fut1,fut2).get();
        }
        return 0;
    }
