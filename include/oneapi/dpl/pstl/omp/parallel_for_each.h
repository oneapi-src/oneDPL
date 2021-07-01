template <class _ExecutionPolicy, class _ForwardIterator, class _Fp>
void
__parallel_for_each(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Fp __f)
{
    dpl::__omp_backend::__parallel_for(std::forward<_ExecutionPolicy>(__exec), __first, __last, __f);
}
