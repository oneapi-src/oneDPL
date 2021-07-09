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
    if (__size == 0)
    {
        return;
    }

    auto __left_it = __xs;
    auto __right_it = __xe;
    bool __is_swapped_left = false, __is_swapped_right = false;
    auto __pivot = *__xs;

    auto __forward_it = __xs + 1;
    while (__forward_it <= __right_it)
    {
        if (__comp(*__forward_it, __pivot))
        {
            __is_swapped_left = true;
            std::iter_swap(__left_it, __forward_it);
            __left_it++;
            __forward_it++;
        }
        else if (__comp(__pivot, *__forward_it))
        {
            __is_swapped_right = true;
            std::iter_swap(__right_it, __forward_it);
            __right_it--;
        }
        else
        {
            __forward_it++;
        }
    }

    if (__size >= __default_chunk_size)
    {
        _PSTL_PRAGMA(omp taskgroup)
        {
            _PSTL_PRAGMA(omp task untied mergeable)
            {
                if (std::distance(__xs, __left_it) > 0 && __is_swapped_left)
                {
                    __parallel_stable_sort_body(__xs, __left_it - 1, __comp);
                }
            }

            _PSTL_PRAGMA(omp task untied mergeable)
            {
                if (std::distance(__right_it, __xe) && __is_swapped_right)
                {
                    __parallel_stable_sort_body(__right_it + 1, __xe, __comp);
                }
            }
        }
    }
    else
    {
        _PSTL_PRAGMA(omp task untied mergeable)
        {
            if (std::distance(__xs, __left_it) > 0 && __is_swapped_left)
            {
                __parallel_stable_sort_body(__xs, __left_it - 1, __comp);
            }

            if (std::distance(__right_it, __xe) && __is_swapped_right)
            {
                __parallel_stable_sort_body(__right_it + 1, __xe, __comp);
            }
        }
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
