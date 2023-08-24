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

#ifndef _ONEDPL_PSTL_OFFLOAD_INTERNAL_MEMORY_IMPL_H
#define _ONEDPL_PSTL_OFFLOAD_INTERNAL_MEMORY_IMPL_H

#ifndef __SYCL_PSTL_OFFLOAD__
#error "__SYCL_PSTL_OFFLOAD__ macro should be defined to include this header"
#endif

#include <execution>

#include <oneapi/dpl/memory>

#include "usm_memory_replacement.h"

namespace std {

template <class _InputIterator, class _ForwardIterator>
_ForwardIterator uninitialized_copy(const execution::parallel_unsequenced_policy&, _InputIterator __first, _InputIterator __last,
                                    _ForwardIterator __result)
{
    return oneapi::dpl::uninitialized_copy(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __result);
}

template <class _InputIterator, class _Size, class _ForwardIterator>
_ForwardIterator uninitialized_copy_n(const execution::parallel_unsequenced_policy&, _InputIterator __first, _Size __n, _ForwardIterator __result)
{
    return oneapi::dpl::uninitialized_copy_n(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __n, __result);
}

template <class _InputIterator, class _ForwardIterator>
_ForwardIterator uninitialized_move(const execution::parallel_unsequenced_policy&, _InputIterator __first, _InputIterator __last,
                                    _ForwardIterator __result)
{
    return oneapi::dpl::uninitialized_move(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __result);
}

template <class _InputIterator, class _Size, class _ForwardIterator>
_ForwardIterator uninitialized_move_n(const execution::parallel_unsequenced_policy&, _InputIterator __first, _Size __n, _ForwardIterator __result)
{
    return oneapi::dpl::uninitialized_move_n(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __n, __result);
}

template <class _ForwardIterator, class _Tp>
void uninitialized_fill(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    oneapi::dpl::uninitialized_fill(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last, __value);
}

template <class _ForwardIterator, class _Size, class _Tp>
_ForwardIterator uninitialized_fill_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _Size __n, const _Tp& __value)
{
    return oneapi::dpl::uninitialized_fill_n(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __n, __value);
}

template <class _ForwardIterator>
void destroy(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last)
{
    oneapi::dpl::destroy(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _ForwardIterator, class _Size>
void destroy_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _Size __n)
{
    oneapi::dpl::destroy_n(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __n);
}

template <class _ForwardIterator>
void uninitialized_default_construct(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last)
{
    oneapi::dpl::uninitialized_default_construct(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _ForwardIterator, class _Size>
void uninitialized_default_construct_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _Size __n)
{
    oneapi::dpl::uninitialized_default_construct_n(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __n);
}

template <class _ForwardIterator>
void uninitialized_value_construct(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last)
{
    oneapi::dpl::uninitialized_value_construct(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __last);
}

template <class _ForwardIterator, class _Size>
void uninitialized_value_construct_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _Size __n)
{
    oneapi::dpl::uninitialized_value_construct_n(::__pstl_offload::__offload_policy_holder.__get_policy(), __first, __n);
}

} // namespace std

#endif // _ONEDPL_PSTL_OFFLOAD_INTERNAL_MEMORY_IMPL_H
