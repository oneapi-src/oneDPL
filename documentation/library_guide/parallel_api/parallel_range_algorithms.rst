Parallel Range Algorithms
#########################

C++20 introduces the Ranges library. C++20 standard splits ranges into two categories: factories and adaptors.
A range factory does not have underlying data. An element is generated on success by an index or by dereferencing an iterator.
A range adaptor, from the |onedpl_long| (|onedpl_short|) perspective, is a utility that transforms the base range,
or another adapted range, into a view with custom behavior.

|onedpl_short| only supports random access ranges, because they allow efficient and constant-time access
to elements at any position in the range. This enables effective workload distribution among multiple threads
or processing units, which is crucial for achieving high performance in parallel execution.

The ``oneapi::dpl::ranges`` namespace supports integration with the Ranges Library coming with the C++20 standard
and introduced by ``std::ranges`` namespace, allowing you to leverage oneDPL parallel algorithms with the standard ranges paradigm.
The functionality is implemented for the host and the device execution policies and requires C++20.

.. Note::

  The use of the ``oneapi::dpl::ranges`` requires C++20 and the C++ standard libraries coming with GCC 10 (or higher) or Clang 10 (or higher).


Supported Range Views
---------------------

The following C++ standard random access adaptors and factories are supported with the oneDPL parallel range algorithms:

* ``std::ranges::views::all``: A range adaptor object that returns a view that includes all elements of its range argument.
* ``std::ranges::iota_view``: A range factory that generates a sequence of N elements, which starts from an initial value and ends by final N-1.
* ``std::ranges::single_view``: A view that contains exactly one element of a specified value.
* ``std::ranges::subrange``: A utility that combines an iterator and a sentinel into a single object that models a view.
* ``std::ranges::transform_view``: A range adaptor that represents a view of a underlying sequence after applying a transformation to each element.
* ``std::ranges::reverse_view``: A range adaptor that produces a reversed sequence of elements provided by another view.
* ``std::ranges::take_view``: A range adaptor that produces a view of the first N elements from another view.
* ``std::ranges::drop_view``: A range adaptor that produces a view excluding the first N elements from another view.

Supported Algorithms
--------------------

The following algorithms are available in the ``namespace oneapi::dpl::ranges`` to use with the mentioned standard ranges:

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

Usage Example for Parallel Range Algorithms
-------------------------------------------

.. code:: cpp

    using namespace oneapi::dpl::ranges;

    {        
        std::vector<int> vec_in = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<int> vec_out(vec_in.size());

        auto view_in = std::ranges::views::all(vec_in) | std::ranges::views::reverse;
        copy(oneapi::dpl::execution::par, view_in, vec_out);
    }
    {
        using shared_allocator = sycl::usm_allocator<int, sycl::usm::alloc::shared>;

        std::vector<int, shared_allocator> vec_in = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<int, shared_allocator> vec_out(vec_in.size());

        auto view_in = std::ranges::subrange(vec_in.begin(), vec_in.end()) | std::ranges::views::reverse;
        copy(oneapi::dpl::execution::dpcpp_default, view_in, std::span(vec_out));
    }
