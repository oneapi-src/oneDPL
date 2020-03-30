// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _PSTL_GLUE_NUMERIC_DEFS_H
#define _PSTL_GLUE_NUMERIC_DEFS_H

#include <iterator>

#include "execution_defs.h"

namespace std
{
// [reduce]

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
       _BinaryOperation __binary_op);

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init);

template <class _ExecutionPolicy, class _ForwardIterator>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                 typename iterator_traits<_ForwardIterator>::value_type>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last);

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                 _ForwardIterator2 __first2, _Tp __init);

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation1,
          class _BinaryOperation2>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                 _ForwardIterator2 __first2, _Tp __init, _BinaryOperation1 __binary_op1,
                 _BinaryOperation2 __binary_op2);

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
                 _BinaryOperation __binary_op, _UnaryOperation __unary_op);

// [exclusive.scan]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
exclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _Tp __init);

#if !_PSTL_EXCLUSIVE_SCAN_WITH_BINARY_OP_AMBIGUITY
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
exclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op);
#else
template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
_ForwardIterator2
exclusive_scan(pstl::execution::sequenced_policy, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op);

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
_ForwardIterator2
exclusive_scan(pstl::execution::unsequenced_policy, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op);

#    if _PSTL_USE_PAR_POLICIES
template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
_ForwardIterator2
exclusive_scan(pstl::execution::parallel_policy, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op);

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
_ForwardIterator2
exclusive_scan(pstl::execution::parallel_unsequenced_policy, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op);
#    endif
#endif

// [inclusive.scan]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result);

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _BinaryOperation __binary_op);

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _BinaryOperation __binary_op, _Tp __init);

// [transform.exclusive.scan]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation,
          class _UnaryOperation>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform_exclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                         _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op,
                         _UnaryOperation __unary_op);

// [transform.inclusive.scan]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation,
          class _UnaryOperation, class _Tp>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform_inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                         _ForwardIterator2 __result, _BinaryOperation __binary_op, _UnaryOperation __unary_op,
                         _Tp __init);

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _UnaryOperation,
          class _BinaryOperation>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform_inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                         _ForwardIterator2 __result, _BinaryOperation __binary_op, _UnaryOperation __unary_op);

// [adjacent.difference]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
adjacent_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                    _ForwardIterator2 __d_first, _BinaryOperation op);

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
pstl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
adjacent_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                    _ForwardIterator2 __d_first);

} // namespace std
#endif /* _PSTL_GLUE_NUMERIC_DEFS_H */
