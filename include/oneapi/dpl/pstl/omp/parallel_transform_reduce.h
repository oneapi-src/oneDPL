template <class _ExecutionPolicy, class _RandomAccessIterator, class _UnaryOp, class _Value, class _Combiner,
    class _Reduction>
_Value
__parallel_transform_reduce(_ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __last,
                            _UnaryOp __unary_op, _Value __init, _Combiner __combiner, _Reduction __reduction)
{

    if (__first == __last)
    {
        return __init;
    }

    _Value __result = __init;
    if (omp_in_parallel())
    {
        // We don't create a nested parallel region in an existing parallel
        // region: just create tasks
        __result = dpl::__omp_backend::__transform_reduce_body(__first, __last, __unary_op, __init, __combiner,
                                                               __reduction);
    }
    else
    {
        // Create a parallel region, and a single thread will create tasks
        // for the region.
        _PSTL_PRAGMA(omp parallel)
        _PSTL_PRAGMA(omp single)
        {
            __result = dpl::__omp_backend::__transform_reduce_body(__first, __last, __unary_op, __init, __combiner,
                                                                   __reduction);
        }
    }

    return __result;
}
