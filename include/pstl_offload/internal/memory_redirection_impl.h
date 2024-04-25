// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PSTL_OFFLOAD_INTERNAL_MEMORY_REDIRECTION_IMPL_H
#define _ONEDPL_PSTL_OFFLOAD_INTERNAL_MEMORY_REDIRECTION_IMPL_H

#if !__SYCL_PSTL_OFFLOAD__
#    error "PSTL offload compiler mode should be enabled to use this header"
#endif

#include <execution>

#include <oneapi/dpl/memory>

#include "usm_memory_replacement.h"

namespace std
{

// All the algorithms below get the policy from static __offload_policy_holder object.
// They needs to be explicitly marked static because, otherwise, function templates behave
// like inline that can result in using only one device in all translation units no matter which
// PSTL offload option argument was used for the particular translation unit compilation

template <class _InputIterator, class _ForwardIterator>
static _ForwardIterator
uninitialized_copy(const execution::parallel_unsequenced_policy&, _InputIterator __first, _InputIterator __last,
                   _ForwardIterator __result)
{
    return oneapi::dpl::uninitialized_copy(
        ::__pstl_offload::__offload_policy_holder_type::__get_policy(::__pstl_offload::__offload_policy_holder),
        __first, __last, __result);
}

template <class _InputIterator, class _Size, class _ForwardIterator>
static _ForwardIterator
uninitialized_copy_n(const execution::parallel_unsequenced_policy&, _InputIterator __first, _Size __n,
                     _ForwardIterator __result)
{
    return oneapi::dpl::uninitialized_copy_n(
        ::__pstl_offload::__offload_policy_holder_type::__get_policy(::__pstl_offload::__offload_policy_holder),
        __first, __n, __result);
}

template <class _InputIterator, class _ForwardIterator>
static _ForwardIterator
uninitialized_move(const execution::parallel_unsequenced_policy&, _InputIterator __first, _InputIterator __last,
                   _ForwardIterator __result)
{
    return oneapi::dpl::uninitialized_move(
        ::__pstl_offload::__offload_policy_holder_type::__get_policy(::__pstl_offload::__offload_policy_holder),
        __first, __last, __result);
}

template <class _InputIterator, class _Size, class _ForwardIterator>
static _ForwardIterator
uninitialized_move_n(const execution::parallel_unsequenced_policy&, _InputIterator __first, _Size __n,
                     _ForwardIterator __result)
{
    return oneapi::dpl::uninitialized_move_n(
        ::__pstl_offload::__offload_policy_holder_type::__get_policy(::__pstl_offload::__offload_policy_holder),
        __first, __n, __result);
}

template <class _ForwardIterator, class _Tp>
static void
uninitialized_fill(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last,
                   const _Tp& __value)
{
    oneapi::dpl::uninitialized_fill(
        ::__pstl_offload::__offload_policy_holder_type::__get_policy(::__pstl_offload::__offload_policy_holder),
        __first, __last, __value);
}

template <class _ForwardIterator, class _Size, class _Tp>
static _ForwardIterator
uninitialized_fill_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _Size __n,
                     const _Tp& __value)
{
    return oneapi::dpl::uninitialized_fill_n(
        ::__pstl_offload::__offload_policy_holder_type::__get_policy(::__pstl_offload::__offload_policy_holder),
        __first, __n, __value);
}

template <class _ForwardIterator>
static void
destroy(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _ForwardIterator __last)
{
    oneapi::dpl::destroy(
        ::__pstl_offload::__offload_policy_holder_type::__get_policy(::__pstl_offload::__offload_policy_holder),
        __first, __last);
}

template <class _ForwardIterator, class _Size>
static void
destroy_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _Size __n)
{
    oneapi::dpl::destroy_n(
        ::__pstl_offload::__offload_policy_holder_type::__get_policy(::__pstl_offload::__offload_policy_holder),
        __first, __n);
}

template <class _ForwardIterator>
static void
uninitialized_default_construct(const execution::parallel_unsequenced_policy&, _ForwardIterator __first,
                                _ForwardIterator __last)
{
    oneapi::dpl::uninitialized_default_construct(
        ::__pstl_offload::__offload_policy_holder_type::__get_policy(::__pstl_offload::__offload_policy_holder),
        __first, __last);
}

template <class _ForwardIterator, class _Size>
static void
uninitialized_default_construct_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _Size __n)
{
    oneapi::dpl::uninitialized_default_construct_n(
        ::__pstl_offload::__offload_policy_holder_type::__get_policy(::__pstl_offload::__offload_policy_holder),
        __first, __n);
}

template <class _ForwardIterator>
static void
uninitialized_value_construct(const execution::parallel_unsequenced_policy&, _ForwardIterator __first,
                              _ForwardIterator __last)
{
    oneapi::dpl::uninitialized_value_construct(
        ::__pstl_offload::__offload_policy_holder_type::__get_policy(::__pstl_offload::__offload_policy_holder),
        __first, __last);
}

template <class _ForwardIterator, class _Size>
static void
uninitialized_value_construct_n(const execution::parallel_unsequenced_policy&, _ForwardIterator __first, _Size __n)
{
    oneapi::dpl::uninitialized_value_construct_n(
        ::__pstl_offload::__offload_policy_holder_type::__get_policy(::__pstl_offload::__offload_policy_holder),
        __first, __n);
}

} // namespace std

#endif // _ONEDPL_PSTL_OFFLOAD_INTERNAL_MEMORY_REDIRECTION_IMPL_H
