namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <typename _F1, typename _F2>
void
__parallel_invoke_body(_F1&& __f1, _F2&& __f2)
{
    _PSTL_PRAGMA(omp taskgroup)
    {
        _PSTL_PRAGMA(omp task) { std::forward<_F1>(__f1)(); }
        _PSTL_PRAGMA(omp task) { std::forward<_F2>(__f2)(); }
    }
}

template <class _ExecutionPolicy, typename _F1, typename _F2>
void
__parallel_invoke(_ExecutionPolicy&&, _F1&& __f1, _F2&& __f2)
{
    if (omp_in_parallel())
    {
        __parallel_invoke_body(std::forward<_F1>(__f1), std::forward<_F2>(__f2));
    }
    else
    {
        _PSTL_PRAGMA(omp parallel)
        _PSTL_PRAGMA(omp single)
        __parallel_invoke_body(std::forward<_F1>(__f1), std::forward<_F2>(__f2));
    }
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
