Range-based API algorithms
##########################

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

The signature example of the range-based algorithms looks like::

  template <typename ExecutionPolicy, typename Range1, typename Range2>
  void copy(ExecutionPolicy&& exec, Range1&& source, Range2&& destination);

where ``source`` is used instead of two iterators to represent the input and ``destination`` represents the output.

These algorithms are declared in ``oneapi::dpl::experimental::ranges`` namespace and implemented only for |dpcpp_long| policies.
In order to make these algorithm available the ``<oneapi/dpl/ranges>`` header should be included (after ``<oneapi/dpl/execution>``).
Use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher) or Clang 7 (or higher).

The following viewable ranges (CPO's) are declared in ``oneapi::dpl::experimental::ranges`` namespace. Only those are 
allowed to use as ranges for range-based algorithms.

* ``views::iota``. A range factory - generates a sequence of N elements which starts from an initial value and ends by final N-1.
* ``views::all``. A custom utility - represents a view of all or a part of ``sycl::buffer`` underlying elements for reading and writing on a device.
* ``views::all_read``. A custom utility - represents a view of all or a part of ``sycl::buffer`` underlying elements for reading on a device.
* ``views::all_write``. A custom utility - represents a view of all or a part of ``sycl::buffer`` underlying elements for writing on a device.
* ``views::host_all``. A custom utility - represents a view of all or a part of ``sycl::buffer`` underlying elements for reading and writing on the host.
* ``views::subrange``. A utility - represents a view of USM data range defined by a two USM pointers.
* ``views::zip``. A custom range adapter - produces one ``zip_view`` from other several views.
* ``views::transform``. A range adapter - represents a view of a underlying sequence after applying a transformation to each element.
* ``views::reverse``. A range adapter - produces a reversed sequence of elements provided by another view.
* ``views::take``. A range adapter - produces a view of the first N elements from another view.
* ``views::drop``. A range adapter - produces a view excluding the first N elements from another view. “

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
