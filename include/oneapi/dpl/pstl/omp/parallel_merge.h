template <class _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
    typename _RandomAccessIterator3, typename _Compare, typename _LeafMerge>
void
__parallel_merge(_ExecutionPolicy&& __exec, _RandomAccessIterator1 __xs, _RandomAccessIterator1 __xe,
                 _RandomAccessIterator2 __ys, _RandomAccessIterator2 __ye, _RandomAccessIterator3 __zs, _Compare __comp,
                 _LeafMerge __leaf_merge)

{
    __serial_backend::__parallel_merge(std::forward<_ExecutionPolicy>(__exec), __xs, __xe, __ys, __ye, __zs, __comp,
                                       __leaf_merge);
}
