Range-Based API Algorithms
##########################

C++20 introduces the Ranges library. C++20 standard splits ranges into two categories: factories and adaptors.
A range factory does not have underlying data. An element is generated on success by an index or by dereferencing an iterator.
A range adaptor, from the |onedpl_long| (|onedpl_short|) perspective, is a utility that transforms the base range,
or another adapted range, into a view with custom behavior.

|onedpl_short| supports just random access ranges, because they allow efficient and constant-time access to elements at any position in the range. This enables effective workload distribution among multiple threads or processing units, which is crucial for achieving high performance in parallel execution.

|onedpl_short| introduces two set of range based algorithms:

* The oneapi::dpl::ext::ranges namespace supports integration with the Ranges Library comming with C++20 standard and introduced by std::ranges namespace, allowing you to leverage oneDPL parallel algorithms with the standard ranges paradigm. It requires C++20.
* The oneapi::dpl::experimental::ranges namespace supports integration with oneDPL ranges introducing by oneapi::dpl::experimental::ranges namespace, allowing you to leverage oneDPL parallel algorithms with the range functionality like the Ranges Library from C++20 standard. The functionality requires C++17.

.. Note::

  The use of the oneapi::dpl::ext::ranges requires C++20 and the C++ standard libraries coming with GCC 10 (or higher) or Clang 10 (or higher).
  The use of the oneapi::dpl::experimental::ranges requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher) or Clang 7 (or higher).


Range-Based API (The standard C++ Ranges Library)
-------------------------------------------------

The following C++ standard random access adaptors and factories are supported with the oneDPL parallel algorithms:

* ``std::ranges::views::all``: A range adaptor object that returns a view that includes all elements of its range argument.
* ``std::ranges::iota_view``: A range factory that generates a sequence of N elements, which starts from an initial value and ends by final N-1.
* ``std::ranges::single_view``: A view that contains exactly one element of a specified value.
* ``std::ranges::subrange``: A utility that combines together an iterator and a sentinel into a single object that models a view.
* ``std::ranges::transform_view``: A range adapter that represents a view of a underlying sequence after applying a transformation to each element.
* ``std::ranges::reverse_view``: A range adapter that produces a reversed sequence of elements provided by another view.
* ``std::ranges::take_view``: A range adapter that produces a view of the first N elements from another view.
* ``std::ranges::drop_view``: A range adapter that produces a view excluding the first N elements from another view.

The following algorithms are available the oneapi::dpl::ext::ranges namespace to use with the metioned standard ranges:

* ``for_each``
* ``transform``
* ``find``
* ``find_if``
* ``find_if_not``
* ``all_of``
* ``any_of``
* ``none_of``
* ``adjacent_find``
* ``search``
* ``search_n``
* ``count_if``
* ``count``
* ``equal``
* ``sort``
* ``stable_sort``
* ``is_sorted``
* ``min_element``
* ``max_element``
* ``copy``
* ``copy_if``
* ``merge``

Experimental Range-Based API (The oneDPL ranges)
------------------------------------------------

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

.. _viewable-ranges:

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
