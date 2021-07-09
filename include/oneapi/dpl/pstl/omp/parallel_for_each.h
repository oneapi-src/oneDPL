namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <class _ExecutionPolicy, class _ForwardIterator, class _Fp>
void
__parallel_for_each(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Fp __f)
{
    //TODO: to implement parallel dpl::__omp_backend::__parallel_for_each
    for (auto __iter = __first; __iter != __last; ++__iter)
        __f(*__iter);
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi

