// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
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

#ifndef _ONEDPL_GLUE_NUMERIC_IMPL_H
#define _ONEDPL_GLUE_NUMERIC_IMPL_H

#include <functional>

#include "utils.h"

#if _ONEDPL_HETERO_BACKEND
#    include "hetero/algorithm_impl_hetero.h"
#    include "hetero/numeric_impl_hetero.h"
#endif

#include "numeric_fwd.h"
#include "execution_impl.h"

namespace oneapi
{
namespace dpl
{

// [reduce]

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
       _BinaryOperation __binary_op)
{
    return transform_reduce(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, __binary_op,
                            oneapi::dpl::__internal::__no_op());
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init)
{
    return transform_reduce(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, ::std::plus<_Tp>(),
                            oneapi::dpl::__internal::__no_op());
}

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy,
                                                      typename ::std::iterator_traits<_ForwardIterator>::value_type>
reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    typedef typename ::std::iterator_traits<_ForwardIterator>::value_type _ValueType;
    return transform_reduce(::std::forward<_ExecutionPolicy>(__exec), __first, __last, _ValueType{},
                            ::std::plus<_ValueType>(), oneapi::dpl::__internal::__no_op());
}

// [transform.reduce]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                 _ForwardIterator2 __first2, _Tp __init)
{
    typedef typename ::std::iterator_traits<_ForwardIterator1>::value_type _InputType;
    return oneapi::dpl::__internal::__pattern_transform_reduce(
        ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __init, ::std::plus<_InputType>(),
        ::std::multiplies<_InputType>(),
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec),
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec));
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation1,
          class _BinaryOperation2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                 _ForwardIterator2 __first2, _Tp __init, _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2)
{
    return oneapi::dpl::__internal::__pattern_transform_reduce(
        ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __init, __binary_op1, __binary_op2,
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec),
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec));
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _Tp>
transform_reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
                 _BinaryOperation __binary_op, _UnaryOperation __unary_op)
{
    return oneapi::dpl::__internal::__pattern_transform_reduce(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __init, __binary_op, __unary_op,
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec),
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec));
}

// [exclusive.scan]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
exclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _Tp __init)
{
    return transform_exclusive_scan(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result, __init,
                                    ::std::plus<_Tp>(), oneapi::dpl::__internal::__no_op());
}

#if !_ONEDPL_EXCLUSIVE_SCAN_WITH_BINARY_OP_AMBIGUITY
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
exclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op)
{
    return transform_exclusive_scan(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result, __init,
                                    __binary_op, oneapi::dpl::__internal::__no_op());
}
#else
template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
_ForwardIterator2
exclusive_scan(oneapi::dpl::execution::sequenced_policy __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op)
{
    return transform_exclusive_scan(__exec, __first, __last, __result, __init, __binary_op,
                                    oneapi::dpl::__internal::__no_op());
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
_ForwardIterator2
exclusive_scan(oneapi::dpl::execution::unsequenced_policy __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op)
{
    return transform_exclusive_scan(__exec, __first, __last, __result, __init, __binary_op,
                                    oneapi::dpl::__internal::__no_op());
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
_ForwardIterator2
exclusive_scan(oneapi::dpl::execution::parallel_policy __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op)
{
    return transform_exclusive_scan(__exec, __first, __last, __result, __init, __binary_op,
                                    oneapi::dpl::__internal::__no_op());
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
_ForwardIterator2
exclusive_scan(oneapi::dpl::execution::parallel_unsequenced_policy __exec, _ForwardIterator1 __first,
               _ForwardIterator1 __last, _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op)
{
    return transform_exclusive_scan(__exec, __first, __last, __result, __init, __binary_op,
                                    oneapi::dpl::__internal::__no_op());
}

#    if _ONEDPL_BACKEND_SYCL
template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation, class... PolicyParams>
_ForwardIterator2
exclusive_scan(const oneapi::dpl::execution::device_policy<PolicyParams...>& __exec, _ForwardIterator1 __first,
               _ForwardIterator1 __last, _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op)
{
    return transform_exclusive_scan(__exec, __first, __last, __result, __init, __binary_op,
                                    oneapi::dpl::__internal::__no_op());
}

#        if _ONEDPL_FPGA_DEVICE
template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation, class KernelName,
          int factor>
_ForwardIterator2
exclusive_scan(const oneapi::dpl::execution::fpga_policy<factor, KernelName>& __exec, _ForwardIterator1 __first,
               _ForwardIterator1 __last, _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op)
{
    return transform_exclusive_scan(__exec, __first, __last, __result, __init, __binary_op,
                                    oneapi::dpl::__internal::__no_op());
}
#        endif // _ONEDPL_FPGA_DEVICE
#    endif     // _ONEDPL_BACKEND_SYCL

#endif // _ONEDPL_EXCLUSIVE_SCAN_WITH_BINARY_OP_AMBIGUITY

// [inclusive.scan]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result)
{
    typedef typename ::std::iterator_traits<_ForwardIterator1>::value_type _InputType;
    return transform_inclusive_scan(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
                                    ::std::plus<_InputType>(), oneapi::dpl::__internal::__no_op());
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _BinaryOperation __binary_op)
{
    return transform_inclusive_scan(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result, __binary_op,
                                    oneapi::dpl::__internal::__no_op());
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _BinaryOperation __binary_op, _Tp __init)
{
    return transform_inclusive_scan(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result, __binary_op,
                                    oneapi::dpl::__internal::__no_op(), __init);
}

// [transform.exclusive.scan]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation,
          class _UnaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform_exclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                         _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op,
                         _UnaryOperation __unary_op)
{
    return oneapi::dpl::__internal::__pattern_transform_scan(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result, __unary_op, __init, __binary_op,
        /*inclusive=*/::std::false_type(),
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec),
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec));
}

// [transform.inclusive.scan]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation,
          class _UnaryOperation, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform_inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                         _ForwardIterator2 __result, _BinaryOperation __binary_op, _UnaryOperation __unary_op,
                         _Tp __init)
{
    return oneapi::dpl::__internal::__pattern_transform_scan(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result, __unary_op, __init, __binary_op,
        /*inclusive=*/::std::true_type(),
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec),
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec));
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _UnaryOperation,
          class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
transform_inclusive_scan(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                         _ForwardIterator2 __result, _BinaryOperation __binary_op, _UnaryOperation __unary_op)
{
    return oneapi::dpl::__internal::__pattern_transform_scan(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result, __unary_op, __binary_op,
        /*inclusive=*/::std::true_type(),
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec),
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec));
}

// [adjacent.difference]

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
adjacent_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                    _ForwardIterator2 __d_first, _BinaryOperation __op)
{

    if (__first == __last)
        return __d_first;

    return oneapi::dpl::__internal::__pattern_adjacent_difference(
        ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __d_first, __op,
        oneapi::dpl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec),
        oneapi::dpl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator1, _ForwardIterator2>(
            __exec));
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator2>
adjacent_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                    _ForwardIterator2 __d_first)
{
    typedef typename ::std::iterator_traits<_ForwardIterator1>::value_type _ValueType;
    return adjacent_difference(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __d_first,
                               ::std::minus<_ValueType>());
}

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_GLUE_NUMERIC_IMPL_H
