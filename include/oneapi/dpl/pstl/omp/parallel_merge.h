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

    // Make sure that the (__xs, __xe] range is always the larger one.
    /* if (__size_y > __size_x) {
        std::swap(__xs, __ys);
        std::swap(__xe, __ye);
        std::swap(__size_x, __size_y);
    } */

    // Split the (__ys, __ye] range into chunks.
    std::size_t __n_chunks{0}, __chunk_size{0}, __first_chunk_size{0};
    __chunk_partitioner(__ys, __ye, __n_chunks, __chunk_size, __first_chunk_size, __default_chunk_size);

    /*
     * Standard parallel merge: for each chunk locate the target offset of each of it's
     * values. Write that value into zs (this assumes that "zs" is a separate container.)
     */
    _PSTL_PRAGMA(omp taskloop)
    for (std::size_t __chunk = 0; __chunk < __n_chunks; ++__chunk)
    {
        auto __this_chunk_size = __chunk == 0 ? __first_chunk_size : __chunk_size;
        auto __index = __chunk == 0 ? 0 : (__chunk * __chunk_size) + (__first_chunk_size - __chunk_size);
        auto __begin = std::next(__ys, __index);
        auto __end = std::next(__begin, __this_chunk_size);

        for (auto __value = __begin; __value < __end; ++__value)
        {
            auto __ys_index = std::distance(__ys, __value);
            auto __xs_index = std::distance(__xs, std::lower_bound(__xs, __xe, *__value, __comp));

            *(__zs + __xs_index + __ys_index) = *__value;
        }
    }
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

    if (omp_in_parallel())
    {
        __parallel_merge_body(__size_x, __size_y, __xs, __xe, __ys, __ye, __zs, __comp, __leaf_merge);
    }
    else
    {
        _PSTL_PRAGMA(omp parallel)
        _PSTL_PRAGMA(omp single)
        __parallel_merge_body(__size_x, __size_y, __xs, __xe, __ys, __ye, __zs, __comp, __leaf_merge);
    }

    /* __serial_backend::__parallel_merge(std::forward<_ExecutionPolicy>(__exec), __xs, __xe, __ys, __ye, __zs, __comp,
                                       __leaf_merge);*/
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
