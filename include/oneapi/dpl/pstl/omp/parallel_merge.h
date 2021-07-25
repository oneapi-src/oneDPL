namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _RandomAccessIterator3,
          typename _Compare, typename _LeafMerge>
void
__parallel_merge_body(std::size_t __size_x, std::size_t __size_y, _RandomAccessIterator1 __xs,
                      _RandomAccessIterator1 __xe, _RandomAccessIterator2 __ys, _RandomAccessIterator2 __ye,
                      _RandomAccessIterator3 __zs, _Compare __comp, _LeafMerge __leaf_merge)
{

    if (__size_x + __size_y <= __default_chunk_size)
    {
        __leaf_merge(__xs, __xe, __ys, __ye, __zs, __comp);
        return;
    }

    _RandomAccessIterator1 __xm;
    _RandomAccessIterator2 __ym;

    if (__size_x < __size_y)
    {
        __ym = __ys + (__size_y / 2);
        __xm = ::std::upper_bound(__xs, __xe, *__ym, __comp);
    }
    else
    {
        __xm = __xs + (__size_x / 2);
        __ym = ::std::lower_bound(__ys, __ye, *__xm, __comp);
    }

    auto __zm = __zs + std::distance(__xs, __xm) + std::distance(__ys, __ym);

    _PSTL_PRAGMA(omp task untied mergeable)
    __parallel_merge_body(std::distance(__xs, __xm), std::distance(__ys, __ym), __xs, __xm, __ys, __ym, __zs, __comp,
                          __leaf_merge);

    _PSTL_PRAGMA(omp task untied mergeable)
    __parallel_merge_body(std::distance(__xm, __xe), std::distance(__ym, __ye), __xm, __xe, __ym, __ye, __zm, __comp,
                          __leaf_merge);
}

template <class _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _RandomAccessIterator3, typename _Compare, typename _LeafMerge>
void
__parallel_merge(_ExecutionPolicy&& /*__exec*/, _RandomAccessIterator1 __xs, _RandomAccessIterator1 __xe,
                 _RandomAccessIterator2 __ys, _RandomAccessIterator2 __ye, _RandomAccessIterator3 __zs, _Compare __comp,
                 _LeafMerge __leaf_merge)

{
    std::size_t __size_x = std::distance(__xs, __xe);
    std::size_t __size_y = std::distance(__ys, __ye);

    // If the size is too small, don't bother parallelizing it.
    if (__size_x + __size_y <= __default_chunk_size)
    {
        __leaf_merge(__xs, __xe, __ys, __ye, __zs, __comp);
        return;
    }

    /*
     * Run the merge in parallel by chunking it up. Use the smaller range (if any) as the iteration range, and the
     * larger range as the search range.
     */

    if (omp_in_parallel())
    {
        __parallel_merge_body(__size_x, __size_y, __xs, __xe, __ys, __ye, __zs, __comp, __leaf_merge);
        _PSTL_PRAGMA(omp barrier)
    }
    else
    {
        _PSTL_PRAGMA(omp parallel)
        {
            _PSTL_PRAGMA(omp single)
            __parallel_merge_body(__size_x, __size_y, __xs, __xe, __ys, __ye, __zs, __comp, __leaf_merge);
            _PSTL_PRAGMA(omp barrier)
        }
    }

    /* __serial_backend::__parallel_merge(std::forward<_ExecutionPolicy>(__exec), __xs, __xe, __ys, __ye, __zs, __comp,
                                       __leaf_merge);*/
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
