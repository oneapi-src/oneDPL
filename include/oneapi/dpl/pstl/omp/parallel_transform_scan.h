namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <class _ExecutionPolicy, class _Index, class _Up, class _Tp, class _Cp, class _Rp, class _Sp>
_Tp
__parallel_transform_scan(_ExecutionPolicy&&, _Index __n, _Up __u, _Tp __init, _Cp __combine, _Rp __brick_reduce,
                          _Sp __scan)
{
    return __scan(_Index(0), __n, __init);
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
