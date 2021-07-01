template <class _ExecutionPolicy, class _Index, class _Up, class _Tp, class _Cp, class _Rp, class _Sp>
_Tp
__parallel_transform_scan(_ExecutionPolicy&& __exec, _Index __n, _Up __u, _Tp __init, _Cp __combine, _Rp __brick_reduce,
                          _Sp __scan)
{

    return __serial_backend::__parallel_transform_scan(std::forward<_ExecutionPolicy>(__exec), __n, __u, __init,
                                                       __combine, __brick_reduce, __scan);
}
