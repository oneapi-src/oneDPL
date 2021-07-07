template <class _Value, typename _ChunkReducer, typename _Reduction>
auto
__parallel_reduce_chunks(std::uint32_t start, std::uint32_t end, _ChunkReducer __reduce_chunk, _Reduction __reduce)
    -> _Value
{
    _Value v1, v2;

    if (end - start == 1)
    {
        return __reduce_chunk(start);
    }
    else if (end - start == 2)
    {
        _PSTL_PRAGMA(omp task shared(v1))
        v1 = __reduce_chunk(start);

        _PSTL_PRAGMA(omp task shared(v2))
        v2 = __reduce_chunk(start + 1);
    }
    else
    {
        auto middle = start + ((end - start) / 2);

        _PSTL_PRAGMA(omp task shared(v1))
        v1 = __parallel_reduce_chunks<_Value>(start, middle, __reduce_chunk, __reduce);

        _PSTL_PRAGMA(omp task shared(v2))
        v2 = __parallel_reduce_chunks<_Value>(middle, end, __reduce_chunk, __reduce);
    }

    _PSTL_PRAGMA(omp taskwait)
    return __reduce(v1, v2);
}

template <class _RandomAccessIterator, class _Value, typename _RealBody, typename _Reduction>
_Value
__parallel_reduce_body(_RandomAccessIterator __first, _RandomAccessIterator __last, _Value __identity,
                       _RealBody __real_body, _Reduction __reduction)
{

    std::size_t __n_chunks{0}, __chunk_size{0}, __first_chunk_size{0};
    __omp_backend::__chunk_partitioner(__first, __last, __n_chunks, __chunk_size, __first_chunk_size);

    auto __reduce_chunk = [&](std::uint32_t __chunk) {
        auto __this_chunk_size = __chunk == 0 ? __first_chunk_size : __chunk_size;
        auto __index = __chunk == 0 ? 0 : (__chunk * __chunk_size) + (__first_chunk_size - __chunk_size);
        auto __begin = __first + __index;
        auto __end = __begin + __this_chunk_size;

        //IMPORTANT: __real_body call does a serial reduction based on an initial value;
        //in case of passing an identity value, a partial result should be explicitly combined
        //with the previous partial reduced value.

        return __real_body(__begin, __end, __identity);
    };

    return __parallel_reduce_chunks<_Value>(0, __n_chunks, __reduce_chunk, __reduction);
}

//------------------------------------------------------------------------
// Notation:
//      r(i,j,init) returns reduction of init with reduction over [i,j)
//      c(x,y) combines values x and y that were the result of r
//------------------------------------------------------------------------

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Value, typename _RealBody, typename _Reduction>
_Value
__parallel_reduce(_ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __last, _Value __identity,
                  _RealBody __real_body, _Reduction __reduction)
{
    if (__first == __last)
        return __identity;

    // Don't bother parallelizing the work if the size is too small.
    if (__last - __first < static_cast<long>(__default_chunk_size))
    {
        return __real_body(__first, __last, __identity);
    }

    // We don't create a nested parallel region in an existing parallel region:
    // just create tasks.
    if (omp_in_parallel())
    {
        return dpl::__omp_backend::__parallel_reduce_body(__first, __last, __identity, __real_body, __reduction);
    }

    // In any case (nested or non-nested) one parallel region is created and only
    // one thread creates a set of tasks.
    _Value __res = __identity;

    _PSTL_PRAGMA(omp parallel)
    _PSTL_PRAGMA(omp single)
    {
        __res = dpl::__omp_backend::__parallel_reduce_body(__first, __last, __identity, __real_body, __reduction);
    }

    return __res;
}
