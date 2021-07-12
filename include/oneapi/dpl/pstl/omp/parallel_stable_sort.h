namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <typename _RandomAccessIterator, typename _Compare>
void
__parallel_stable_sort_body(_RandomAccessIterator __xs, _RandomAccessIterator __xe, _Compare __comp)
{
    std::size_t __size = std::distance(__xs, __xe);

    if (__size <= __default_chunk_size)
    {
        if (__size > 1)
        {
            auto __mid = __xs + (__size / 2);
            __parallel_stable_sort_body(__xs, __mid, __comp);
            __parallel_stable_sort_body(__mid, __xe, __comp);
            std::inplace_merge(__xs, __mid, __xe, __comp);
        }
    }
    else
    {
        auto __mid = __xs + (__size / 2);
        _PSTL_PRAGMA(omp taskgroup)
        {
            _PSTL_PRAGMA(omp task untied mergeable) { __parallel_stable_sort_body(__xs, __mid, __comp); }
            _PSTL_PRAGMA(omp task untied mergeable) { __parallel_stable_sort_body(__mid, __xe, __comp); }
        }
        std::inplace_merge(__xs, __mid, __xe, __comp);
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
            __parallel_stable_sort_body(__xs, __xe, __comp);
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
            __parallel_stable_sort_body(__xs, __xe, __comp);
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
