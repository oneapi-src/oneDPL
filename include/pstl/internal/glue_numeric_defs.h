/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#ifndef __PSTL_glue_numeric_defs_H
#define __PSTL_glue_numeric_defs_H

#include "execution_defs.h"

namespace std {
// [reduce]

template<class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _Tp>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init, _BinaryOperation __binary_op);

template<class _ExecutionPolicy, class _ForwardIterator, class _Tp>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _Tp>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init);

template<class _ExecutionPolicy, class _ForwardIterator>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, typename iterator_traits<_ForwardIterator>::value_type>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Tp __init);

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Tp __init,
                 _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2);

template<class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init, _BinaryOperation __binary_op, _UnaryOperation __unary_op);

// [exclusive.scan]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy,_ForwardIterator2>
exclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result, _Tp __init);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
_ForwardIterator2 exclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result, _Tp __init,
                                _BinaryOperation __binary_op);

// [inclusive.scan]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy,_ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy,_ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result, _BinaryOperation __binary_op);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy,_ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result, _BinaryOperation __binary_op, _Tp __init);

// [transform.exclusive.scan]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation, class _UnaryOperation>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy,_ForwardIterator2>
transform_exclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result, _Tp __init,
                         _BinaryOperation __binary_op, _UnaryOperation __unary_op);

// [transform.inclusive.scan]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation, class _UnaryOperation, class _Tp>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy,_ForwardIterator2>
transform_inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result, _BinaryOperation __binary_op,
                         _UnaryOperation __unary_op, _Tp __init);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _UnaryOperation, class _BinaryOperation>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy,_ForwardIterator2>
transform_inclusive_scan(_ExecutionPolicy&& __exec,  _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result, _BinaryOperation __binary_op,
                         _UnaryOperation __unary_op);

// [adjacent.difference]

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
adjacent_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __d_first, _BinaryOperation op);

template<class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
pstl::internal::enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
adjacent_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __d_first);

} // namespace std
#endif /* __PSTL_glue_numeric_defs_H */
