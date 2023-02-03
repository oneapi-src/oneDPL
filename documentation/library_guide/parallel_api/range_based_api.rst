Range-Based API Algorithms
##########################
.. Note::

  The use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher)
  or Clang 7 (or higher).

C++20 introduces the Ranges library. C++20 standard splits ranges into two categories: factories and adaptors.
A range factory does not have underlying data. An element is generated on success by an index or by dereferencing an iterator.
A range adaptor, from the |onedpl_long| (|onedpl_short|) perspective, is a utility that transforms the base range,
or another adapted range, into a view with custom behavior.

|onedpl_short| supports an ``iota_view`` range factory.

A ``sycl::buffer`` wrapped with ``all_view`` can be used as the range.

|onedpl_short| considers the supported factories and ``all_view`` as base ranges.
The range adaptors may be combined into a pipeline with a ``base`` range at the beginning. For example:

.. code:: cpp

    sycl::buffer<int> buf(data, sycl::range<1>(10));
    auto range_1 = iota_view(0, 10) | views::reverse();
    auto range_2 = all_view(buf) | views::reverse();

For the range, based on the ``all_view`` factory, data access is permitted on a device only. ``size()`` and ``empty()`` methods are allowed 
to be called on both host and device.

The following algorithms are available to use with the ranges:

* ``adjacent_find``
* ``all_of``
* ``any_of``
* ``copy``
* ``copy_if``
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
* ``none_of``
* ``reduce``
* ``reduce_by_segment``
* ``remove``
* ``remove_if``
* ``remove_copy``
* ``remove_copy_if``
* ``replace``
* ``replace_if``
* ``replace_copy``
* ``replace_copy_if``
* ``reverse``
* ``reverse_copy``
* ``rotate_copy``
* ``search``
* ``sort``
* ``stable_sort``
* ``swap_ranges``
* ``transform``
* ``transform_reduce``
* ``transform_exclusive_scan``
* ``transform_inclusive_scan``
* ``unique``
* ``unique_copy``

The signature example of the range-based algorithms looks like:

.. code:: cpp

   template <typename ExecutionPolicy, typename Range1, typename Range2>
   void copy(ExecutionPolicy&& exec, Range1&& source, Range2&& destination);

where ``source`` is used instead of two iterators to represent the input, and ``destination`` represents the output.

These algorithms are declared in the ``oneapi::dpl::experimental::ranges`` namespace and implemented only for device execution policies.
To make these algorithms available, the ``<oneapi/dpl/ranges>`` header should be included (after ``<oneapi/dpl/execution>``).
Use of the range-based API requires C++17 and the C++ standard libraries that come with GCC 8.1 (or higher) or Clang 7 (or higher).

The following viewable ranges are declared in the ``oneapi::dpl::experimental::ranges`` namespace.
Only the ranges shown below and ``sycl::buffer`` are available as ranges for range-based algorithms.

* ``views::iota``: A range factory that generates a sequence of N elements, which starts from an initial value and ends by final N-1.
* ``views::all``: A custom utility that represents a view of all or a part of ``sycl::buffer`` underlying elements for reading and writing on a device.
* ``views::all_read``: A custom utility that represents a view of all or a part of ``sycl::buffer`` underlying elements for reading on a device.
* ``views::all_write``: A custom utility that represents a view of all or a part of ``sycl::buffer`` underlying elements for writing on a device.
* ``views::host_all``: A custom utility that represents a view of all or a part of ``sycl::buffer`` underlying elements for reading and writing on the host.
* ``views::subrange``: A utility that represents a view of unified shared memory (USM) data range defined by a two USM pointers.
* ``views::zip``: A custom range adapter that produces one ``zip_view`` from other several views.
* ``views::transform``: A range adapter that represents a view of a underlying sequence after applying a transformation to each element.
* ``views::reverse``: A range adapter that produces a reversed sequence of elements provided by another view.
* ``views::take``: A range adapter that produces a view of the first N elements from another view.
* ``views::drop``: A range adapter that produces a view excluding the first N elements from another view.

Example of Range-Based API Usage
--------------------------------

.. code:: cpp

    using namespace oneapi::dpl::experimental::ranges;

    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));
        sycl::buffer<int> B(data2, sycl::range<1>(max_n));

        auto view = all_view(A) | views::reverse();
        auto range_res = all_view<int, sycl::access::mode::write>(B);

        copy(oneapi::dpl::execution::dpcpp_default, view, range_res);
    }
