template <class _RandomAccessIterator, class _Fp>
void
__parallel_for_body(_RandomAccessIterator __first, _RandomAccessIterator __last, _Fp __f)
{
    std::size_t __n_chunks{0}, __chunk_size{0}, __first_chunk_size{0};
    __chunk_partitioner(__first, __last, __n_chunks, __chunk_size, __first_chunk_size);

    // To avoid over-subscription we use taskloop for the nested parallelism
    _PSTL_PRAGMA(omp taskloop)
    for (std::size_t __chunk = 0; __chunk < __n_chunks; ++__chunk)
    {
        auto __this_chunk_size = __chunk == 0 ? __first_chunk_size : __chunk_size;
        auto __index = __chunk == 0 ? 0 : (__chunk * __chunk_size) + (__first_chunk_size - __chunk_size);
        auto __begin = std::next(__first, __index);
        auto __end = std::next(__begin, __this_chunk_size);
        __f(__begin, __end);
    }
}

//------------------------------------------------------------------------
// Notation:
// Evaluation of brick f[i,j) for each subrange [i,j) of [first, last)
//------------------------------------------------------------------------

template <class _ExecutionPolicy, class _RandomAccessIterator, class _Fp>
void
__parallel_for(_ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __last, _Fp __f)
{
    if (omp_in_parallel())
    {
        // we don't create a nested parallel region in an existing parallel
        // region: just create tasks
        dpl::__omp_backend::__parallel_for_body(__first, __last, __f);
    }
    else
    {
        // in any case (nested or non-nested) one parallel region is created and
        // only one thread creates a set of tasks
        _PSTL_PRAGMA(omp parallel)
        _PSTL_PRAGMA(omp single) { dpl::__omp_backend::__parallel_for_body(__first, __last, __f); }
    }
}
