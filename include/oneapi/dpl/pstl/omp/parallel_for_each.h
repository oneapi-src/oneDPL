namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <class _ForwardIterator, class _Fp>
void
__parallel_for_each_body(_ForwardIterator __first, _ForwardIterator __last, _Fp __f)
{
    _PSTL_PRAGMA(omp taskgroup)
    {
        for (auto __iter = __first; __iter != __last; ++__iter)
        {
            _PSTL_PRAGMA(omp task firstprivate(__iter, __f)) { __f(*__iter); }
        }
    }
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Fp>
void
__parallel_for_each(_ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last, _Fp __f)
{
    if (omp_in_parallel())
    {
        // we don't create a nested parallel region in an existing parallel
        // region: just create tasks
        dpl::__omp_backend::__parallel_for_each_body(__first, __last, __f);
    }
    else
    {
        // in any case (nested or non-nested) one parallel region is created and
        // only one thread creates a set of tasks
        _PSTL_PRAGMA(omp parallel)
        _PSTL_PRAGMA(omp single) { dpl::__omp_backend::__parallel_for_each_body(__first, __last, __f); }
    }
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
