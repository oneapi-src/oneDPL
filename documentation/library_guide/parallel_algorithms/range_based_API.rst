Algorithms with range-based API, and supporting classes
#######################################################

Range-based API
---------------

C++20 indroduces the *Ranges* library. С++20 standard splits ranges into two categories: *factories* and *adaptors*.
A range factory doesn't have underlying data. An element is generated on success by an index or by dereferencing an iterator.
A range adaptor, from the DPC++ library perspective, is an utility that transforms *base range*, or another adapted range into 
a view with custom behavior.

The DPC++ library supports ``iota_view`` range factory.

``sycl::buffer`` wrapped with ``all_view`` can be used as the range.

The DPC++ library considers the supported factories and ``all_view`` as base ranges.
The range adaptors may be combined into a pipeline with a ``base`` range at the beginning. For example:

.. code:: cpp

    cl::sycl::buffer<int> buf(data, cl::sycl::range<1>(10));
    auto range_1 = iota_view(0, 10) | views::reverse();
    auto range_2 = all_view(buf) | views::reverse();

For the range, based on the ``all_view`` factory, data access is permitted on a device only. ``size()`` and ``empty()`` methods are allowed 
to be called on both host and device.

The following algorithms are available to use with the ranges:

``for_each``, ``copy``, ``transform``, ``find``, ``find_if``, ``find_if_not``, ``find_end``, ``find_first_of``, ``search``, ``is_sorted``,
``is_sorted_until``, ``reduce``, ``transform_reduce``, ``min_element``, ``max_element``, ``minmax_element``,
``exclusive_scan``, ``inclusive_scan``, ``transform_exclusive_scan``, ``transform_inclusive_scan``.

The signature example of the range-based algorithms looks like:

.. code:: cpp

  template <typename ExecutionPolicy, typename Range1, typename Range2>
  void copy(ExecutionPolicy&& exec, Range1&& source, Range2&& destination);

where ``source`` is used instead of two iterators to represent the input. ``destination`` represents the output.

These algorithms are declared in ``oneapi::dpl::experimental::ranges`` namespace and implemented only for DPC++ policies.
In order to make these algorithm available the ``<oneapi/dpl/ranges>`` header should be included.
Use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher) or Clang 7 (or higher).

The following viewable ranges are declared in ``oneapi::dpl::experimental::ranges`` namespace. Only those are allowed to use as ranges for range-based algorithms.

* ``iota_view``. A range factory - generates a sequence of N elements which starts from an initial value and ends by final N-1.
* ``all_view``. A custom utility - represents a view of all or a part of ``sycl::buffer`` underlying elements.
* ``guard_view``. A custom utility - represents a view of USM data range defined by a two USM pointers.
* ``zip_view``. A custom range adapter - produces one ``zip_view`` from other several views.
* ``transform_view``. A range adapter - represents a view of a underlying sequence after applying a transformation to each element.
* ``reverse_view``. A range adapter - produces a reversed sequence of elements provided by another view.
* ``take_view``. A range adapter - produces a view of the first N elements from another view.
* ``drop_view``. A range adapter - produces a view excluding the first N elements from another view.

Example of Range-based API usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

    using namespace oneapi::dpl::experimental::ranges;

    {
        cl::sycl::buffer<int> A(data, cl::sycl::range<1>(max_n));
        cl::sycl::buffer<int> B(data2, cl::sycl::range<1>(max_n));

        auto view = all_view(A) | views::reverse();
        auto range_res = all_view<int, cl::sycl::access::mode::write>(B);

        copy(oneapi::dpl::execution::dpcpp_default, view, range_res);
    }