namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <class _RandomAccessIterator, class _Value, typename _RealBody, typename _Reduction>
_Value
__parallel_reduce_body(_RandomAccessIterator __first, _RandomAccessIterator __last, _Value __identity,
                       _RealBody __real_body, _Reduction __reduce)
{
    if (__should_run_serial(__first, __last))
    {
        return __real_body(__first, __last, __identity);
    }

    auto __middle = __first + ((__last - __first) / 2);
    _Value __v1(__identity), __v2(__identity);
    __parallel_invoke_body(
        [&]() { __v1 = __parallel_reduce_body(__first, __middle, __identity, __real_body, __reduce); },
        [&]() { __v2 = __parallel_reduce_body(__middle, __last, __identity, __real_body, __reduce); });

    return __reduce(__v1, __v2);
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
    if (__should_run_serial(__first, __last))
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

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
