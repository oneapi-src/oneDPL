// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PSTL_OFFLOAD_INTERNAL_NUMERIC_REDIRECTION_IMPL_H
#define _ONEDPL_PSTL_OFFLOAD_INTERNAL_NUMERIC_REDIRECTION_IMPL_H

#if !__SYCL_PSTL_OFFLOAD__
#    error "PSTL offload compiler mode should be enabled to use this header"
#endif

#include <execution>

#include <oneapi/dpl/numeric>

#include "usm_memory_replacement.h"

namespace std
{

// All the algorithms below get the policy from static __offload_policy_holder object.
// They needs to be explicitly marked static because, otherwise, function templates behave
// like inline that can result in using only one device in all translation units no matter which
// PSTL offload option argument was used for the particular translation unit compilation

template <class _ForwardIterator, class _Tp, class _BinaryOperation>
static _Tp
reduce(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
       _BinaryOperation __binary_op)
{
    return oneapi::dpl::reduce(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __init,
                               __binary_op);
}

template <class _ForwardIterator, class _Tp>
static _Tp
reduce(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, _Tp __init)
{
    return oneapi::dpl::reduce(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __init);
}

template <class _ForwardIterator>
static typename iterator_traits<_ForwardIterator>::value_type
reduce(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last)
{
    return oneapi::dpl::reduce(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp>
static _Tp
transform_reduce(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                 _ForwardIterator2 __first2, _Tp __init)
{
    return oneapi::dpl::transform_reduce(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1,
                                         __first2, __init);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2>
static _Tp
transform_reduce(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                 _ForwardIterator2 __first2, _Tp __init, _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2)
{
    return oneapi::dpl::transform_reduce(::__pstl_offload::__offload_policy_holder.__get_policy(), __first1, __last1,
                                         __first2, __init, __binary_op1, __binary_op2);
}

template <class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation>
static _Tp
transform_reduce(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last,
                 _Tp __init, _BinaryOperation __binary_op, _UnaryOperation __unary_op)
{
    return oneapi::dpl::transform_reduce(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last,
                                         __init, __binary_op, __unary_op);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp>
static _ForwardIterator2
exclusive_scan(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __d_first, _Tp __init)
{
    return oneapi::dpl::exclusive_scan(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last,
                                       __d_first, __init);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
static _ForwardIterator2
exclusive_scan(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __d_first, _Tp __init, _BinaryOperation __binary_op)
{
    return oneapi::dpl::exclusive_scan(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last,
                                       __d_first, __init, __binary_op);
}

template <class _ForwardIterator1, class _ForwardIterator2>
static _ForwardIterator2
inclusive_scan(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result)
{
    return oneapi::dpl::inclusive_scan(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last,
                                       __result);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation>
static _ForwardIterator2
inclusive_scan(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _BinaryOperation __binary_op)
{
    return oneapi::dpl::inclusive_scan(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last,
                                       __result, __binary_op);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation>
static _ForwardIterator2
inclusive_scan(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
               _ForwardIterator2 __result, _BinaryOperation __binary_op, _Tp __init)
{
    return oneapi::dpl::inclusive_scan(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last,
                                       __result, __binary_op, __init);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation, class _UnaryOperation>
static _ForwardIterator2
transform_exclusive_scan(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first,
                         _ForwardIterator1 __last, _ForwardIterator2 __result, _Tp __init, _BinaryOperation __binary_op,
                         _UnaryOperation __unary_op)
{
    return oneapi::dpl::transform_exclusive_scan(::__pstl_offload::__offload_policy_holder.__get_policy(),
                                                 __first, __last, __result, __init, __binary_op, __unary_op);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation, class _UnaryOperation, class _Tp>
static _ForwardIterator2
transform_inclusive_scan(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first,
                         _ForwardIterator1 __last, _ForwardIterator2 __result, _BinaryOperation __binary_op,
                         _UnaryOperation __unary_op, _Tp __init)
{
    return oneapi::dpl::transform_inclusive_scan(::__pstl_offload::__offload_policy_holder.__get_policy(),
                                                 __first, __last, __result, __binary_op, __unary_op, __init);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _UnaryOperation, class _BinaryOperation>
static _ForwardIterator2
transform_inclusive_scan(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first,
                         _ForwardIterator1 __last, _ForwardIterator2 __result, _BinaryOperation __binary_op,
                         _UnaryOperation __unary_op)
{
    return oneapi::dpl::transform_inclusive_scan(::__pstl_offload::__offload_policy_holder.__get_policy(),
                                                 __first, __last, __result, __binary_op, __unary_op);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryOperation>
static _ForwardIterator2
adjacent_difference(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                    _ForwardIterator2 __d_first, _BinaryOperation __op)
{
    return oneapi::dpl::adjacent_difference(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last,
                                            __d_first, __op);
}

template <class _ForwardIterator1, class _ForwardIterator2>
static _ForwardIterator2
adjacent_difference(const execution::parallel_unsequenced_policy&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                    _ForwardIterator2 __d_first)
{
    return oneapi::dpl::adjacent_difference(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last,
                                            __d_first);
}

} // namespace std

#endif // _ONEDPL_PSTL_OFFLOAD_INTERNAL_NUMERIC_REDIRECTION_IMPL_H
