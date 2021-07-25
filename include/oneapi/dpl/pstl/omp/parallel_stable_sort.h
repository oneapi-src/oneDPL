namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

namespace __sort_details
{
template <typename Iterator1, typename Iterator2>
struct __move_value
{
    void
    operator()(Iterator1 __x, Iterator2 __z)
    {
        *__z = ::std::move(*__x);
    }
};

template <typename Iterator1, typename Iterator2>
Iterator2
__parallel_move_range(Iterator1 __first1, Iterator1 __last1, Iterator2 __first2)
{
    std::size_t __size = std::distance(__first1, __last1);

    // Perform serial moving of small chunks

    if (__size <= __default_chunk_size)
    {
        return ::std::move(__first1, __last1, __first2);
    }

    // Perform parallel moving of larger chunks

    std::size_t __n_chunks{0}, __chunk_size{0}, __first_chunk_size{0};
    __chunk_partitioner(__first1, __last1, __n_chunks, __chunk_size, __first_chunk_size);

    _PSTL_PRAGMA(omp taskloop)
    for (std::size_t __chunk = 0; __chunk < __n_chunks; ++__chunk)
    {
        auto __this_chunk_size = __chunk == 0 ? __first_chunk_size : __chunk_size;
        auto __index = __chunk == 0 ? 0 : (__chunk * __chunk_size) + (__first_chunk_size - __chunk_size);
        auto __begin = __first1 + __index;
        auto __end = __begin + __this_chunk_size;
        auto __output = __first2 + __index;
        ::std::move(__begin, __end, __output);
    }

    return __first2 + __size;
}

template <typename Iterator1, typename Iterator2>
struct __move_range
{
    Iterator2
    operator()(Iterator1 __first1, Iterator1 __last1, Iterator2 __first2)
    {
        return __parallel_move_range(__first1, __last1, __first2);
    }
};
} // namespace __sort_details

template <typename _RandomAccessIterator, typename _Compare, typename _LeafSort>
void
__parallel_stable_sort_body(_RandomAccessIterator __xs, _RandomAccessIterator __xe, _Compare __comp,
                            _LeafSort __leaf_sort)
{
    using _ValueType = typename std::iterator_traits<_RandomAccessIterator>::value_type;
    using _VecType = typename std::vector<_ValueType>;
    using _OutputIterator = typename _VecType::iterator;
    using _MoveValueType = typename __omp_backend::__sort_details::__move_value<_RandomAccessIterator, _OutputIterator>;
    using _MoveRangeType = __omp_backend::__sort_details::__move_range<_RandomAccessIterator, _OutputIterator>;

    std::size_t __size = std::distance(__xs, __xe);

    if (__size <= __default_chunk_size)
    {
        __leaf_sort(__xs, __xe, __comp);
    }
    else
    {
        auto __mid = __xs + (__size / 2);
        _PSTL_PRAGMA(omp taskgroup)
        {
            _PSTL_PRAGMA(omp task untied mergeable) { __parallel_stable_sort_body(__xs, __mid, __comp, __leaf_sort); }
            _PSTL_PRAGMA(omp task untied mergeable) { __parallel_stable_sort_body(__mid, __xe, __comp, __leaf_sort); }
        }

        std::inplace_merge(__xs, __mid, __xe, __comp);

#if 0
        // The parallel merge fails some tests. Not sure why...

        // Perform a parallel merge of the sorted ranges into __output.
        _VecType __output(__size);
        _MoveValueType __move_value;
        _MoveRangeType __move_range;
        __utils::__serial_move_merge __merge(__size);
        __parallel_merge_body(
            std::distance(__xs, __mid), std::distance(__mid, __xe), __xs, __mid, __mid, __xe, __output.begin(), __comp,
            [&__merge, &__move_value, &__move_range](_RandomAccessIterator __as, _RandomAccessIterator __ae,
                                                     _RandomAccessIterator __bs, _RandomAccessIterator __be,
                                                     _OutputIterator __cs, _Compare __comp)
            { __merge(__as, __ae, __bs, __be, __cs, __comp, __move_value, __move_value, __move_range, __move_range); });

        // Move the values from __output back in the original source range.
        __sort_details::__parallel_move_range(__output.begin(), __output.end(), __xs);
#endif
    }
}

template <class _ExecutionPolicy, typename _RandomAccessIterator, typename _Compare, typename _LeafSort>
void
__parallel_stable_sort(_ExecutionPolicy&& __exec, _RandomAccessIterator __xs, _RandomAccessIterator __xe,
                       _Compare __comp, _LeafSort __leaf_sort, std::size_t __nsort = 0)
{
    if (__xs >= __xe)
    {
        return;
    }

    if (__nsort <= __default_chunk_size)
    {
        __serial_backend::__parallel_stable_sort(std::forward<_ExecutionPolicy>(__exec), __xs, __xe, __comp,
                                                 __leaf_sort, __nsort);
        return;
    }

    std::size_t __count = static_cast<std::size_t>(std::distance(__xs, __xe));

    if (omp_in_parallel())
    {
        if (__count <= __nsort)
        {
            __parallel_stable_sort_body(__xs, __xe, __comp, __leaf_sort);
        }
        else
        {
            __parallel_stable_partial_sort(__xs, __xe, __comp, __leaf_sort, __nsort);
        }
    }
    else
    {
        _PSTL_PRAGMA(omp parallel)
        _PSTL_PRAGMA(omp single)
        if (__count <= __nsort)
        {
            __parallel_stable_sort_body(__xs, __xe, __comp, __leaf_sort);
        }
        else
        {
            __parallel_stable_partial_sort(__xs, __xe, __comp, __leaf_sort, __nsort);
        }
    }
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
