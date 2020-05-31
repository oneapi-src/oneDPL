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

#ifndef _PSTL_GLUE_MEMORY_IMPL_H
#define _PSTL_GLUE_MEMORY_IMPL_H

#include "execution_defs.h"
#include "utils.h"

#if _PSTL_HETERO_BACKEND
#    include "hetero/algorithm_impl_hetero.h"
#endif

#include "memory_fwd.h"
#include "algorithm_fwd.h"

#include "execution_impl.h"

namespace std
{

// [uninitialized.copy]

template <class _ExecutionPolicy, class _InputIterator, class _ForwardIterator>
dpstd::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_copy(_ExecutionPolicy&& __exec, _InputIterator __first, _InputIterator __last, _ForwardIterator __result)
{
    typedef typename iterator_traits<_InputIterator>::value_type _ValueType1;
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType2;
    typedef typename iterator_traits<_InputIterator>::reference _ReferenceType1;
    typedef typename iterator_traits<_ForwardIterator>::reference _ReferenceType2;
    typedef typename std::decay<_ExecutionPolicy>::type _DecayedExecutionPolicy;

    const auto __is_parallel =
        dpstd::__internal::__is_parallelization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);
    const auto __is_vector =
        dpstd::__internal::__is_vectorization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);

    return dpstd::__internal::__invoke_if_else(
        std::integral_constant < bool, std::is_trivial<_ValueType1>::value&& std::is_trivial<_ValueType2>::value > (),
        [&]() {
            return dpstd::__internal::__pattern_walk2_brick(
                std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
                dpstd::__internal::__brick_copy<_DecayedExecutionPolicy>{}, __is_parallel);
        },
        [&]() {
            return dpstd::__internal::__pattern_walk2(
                std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
                dpstd::__internal::__op_uninitialized_copy<_DecayedExecutionPolicy>{}, __is_vector, __is_parallel);
        });
}

template <class _ExecutionPolicy, class _InputIterator, class _Size, class _ForwardIterator>
dpstd::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_copy_n(_ExecutionPolicy&& __exec, _InputIterator __first, _Size __n, _ForwardIterator __result)
{
    typedef typename iterator_traits<_InputIterator>::value_type _ValueType1;
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType2;
    typedef typename iterator_traits<_InputIterator>::reference _ReferenceType1;
    typedef typename iterator_traits<_ForwardIterator>::reference _ReferenceType2;
    typedef typename std::decay<_ExecutionPolicy>::type _DecayedExecutionPolicy;

    const auto __is_parallel =
        dpstd::__internal::__is_parallelization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);
    const auto __is_vector =
        dpstd::__internal::__is_vectorization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);

    return dpstd::__internal::__invoke_if_else(
        std::integral_constant < bool, std::is_trivial<_ValueType1>::value&& std::is_trivial<_ValueType2>::value > (),
        [&]() {
            return dpstd::__internal::__pattern_walk2_brick_n(
                std::forward<_ExecutionPolicy>(__exec), __first, __n, __result,
                dpstd::__internal::__brick_copy_n<_DecayedExecutionPolicy>{}, __is_parallel);
        },
        [&]() {
            return dpstd::__internal::__pattern_walk2_n(
                std::forward<_ExecutionPolicy>(__exec), __first, __n, __result,
                dpstd::__internal::__op_uninitialized_copy<_DecayedExecutionPolicy>{}, __is_vector, __is_parallel);
        });
}

// [uninitialized.move]

template <class _ExecutionPolicy, class _InputIterator, class _ForwardIterator>
dpstd::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_move(_ExecutionPolicy&& __exec, _InputIterator __first, _InputIterator __last, _ForwardIterator __result)
{
    typedef typename iterator_traits<_InputIterator>::value_type _ValueType1;
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType2;
    typedef typename iterator_traits<_InputIterator>::reference _ReferenceType1;
    typedef typename iterator_traits<_ForwardIterator>::reference _ReferenceType2;
    typedef typename std::decay<_ExecutionPolicy>::type _DecayedExecutionPolicy;

    const auto __is_parallel =
        dpstd::__internal::__is_parallelization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);
    const auto __is_vector =
        dpstd::__internal::__is_vectorization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);

    return dpstd::__internal::__invoke_if_else(
        std::integral_constant < bool, std::is_trivial<_ValueType1>::value&& std::is_trivial<_ValueType2>::value > (),
        [&]() {
            return dpstd::__internal::__pattern_walk2_brick(
                std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
                dpstd::__internal::__brick_copy<_DecayedExecutionPolicy>{}, __is_parallel);
        },
        [&]() {
            return dpstd::__internal::__pattern_walk2(
                std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
                dpstd::__internal::__op_uninitialized_move<_DecayedExecutionPolicy>{}, __is_vector, __is_parallel);
        });
}

template <class _ExecutionPolicy, class _InputIterator, class _Size, class _ForwardIterator>
dpstd::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_move_n(_ExecutionPolicy&& __exec, _InputIterator __first, _Size __n, _ForwardIterator __result)
{
    typedef typename iterator_traits<_InputIterator>::value_type _ValueType1;
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType2;
    typedef typename iterator_traits<_InputIterator>::reference _ReferenceType1;
    typedef typename iterator_traits<_ForwardIterator>::reference _ReferenceType2;
    typedef typename std::decay<_ExecutionPolicy>::type _DecayedExecutionPolicy;

    const auto __is_parallel =
        dpstd::__internal::__is_parallelization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);
    const auto __is_vector =
        dpstd::__internal::__is_vectorization_preferred<_ExecutionPolicy, _InputIterator, _ForwardIterator>(__exec);

    return dpstd::__internal::__invoke_if_else(
        std::integral_constant < bool, std::is_trivial<_ValueType1>::value&& std::is_trivial<_ValueType2>::value > (),
        [&]() {
            return dpstd::__internal::__pattern_walk2_brick_n(
                std::forward<_ExecutionPolicy>(__exec), __first, __n, __result,
                dpstd::__internal::__brick_copy_n<_DecayedExecutionPolicy>{}, __is_parallel);
        },
        [&]() {
            return dpstd::__internal::__pattern_walk2_n(
                std::forward<_ExecutionPolicy>(__exec), __first, __n, __result,
                dpstd::__internal::__op_uninitialized_move<_DecayedExecutionPolicy>{}, __is_vector, __is_parallel);
        });
}

// [uninitialized.fill]

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
dpstd::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_fill(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    typedef typename iterator_traits<_ForwardIterator>::reference _ReferenceType;
    typedef typename std::decay<_ExecutionPolicy>::type _DecayedExecutionPolicy;

    const auto __is_parallel =
        dpstd::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector =
        dpstd::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    dpstd::__internal::__invoke_if_else(
        std::is_arithmetic<_ValueType>(),
        [&]() {
            dpstd::__internal::__pattern_walk_brick(
                std::forward<_ExecutionPolicy>(__exec), __first, __last,
                dpstd::__internal::__brick_fill<_ValueType, _DecayedExecutionPolicy>{_ValueType(__value)},
                __is_parallel);
        },
        [&]() {
            dpstd::__internal::__pattern_walk1(
                std::forward<_ExecutionPolicy>(__exec), __first, __last,
                dpstd::__internal::__op_uninitialized_fill<_Tp, _DecayedExecutionPolicy>{__value}, __is_vector,
                __is_parallel);
        });
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Tp>
dpstd::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_fill_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n, const _Tp& __value)
{
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    typedef typename iterator_traits<_ForwardIterator>::reference _ReferenceType;
    typedef typename std::decay<_ExecutionPolicy>::type _DecayedExecutionPolicy;

    const auto __is_parallel =
        dpstd::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector =
        dpstd::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    return dpstd::__internal::__invoke_if_else(
        std::is_arithmetic<_ValueType>(),
        [&]() {
            return dpstd::__internal::__pattern_walk_brick_n(
                std::forward<_ExecutionPolicy>(__exec), __first, __n,
                dpstd::__internal::__brick_fill_n<_ValueType, _DecayedExecutionPolicy>{_ValueType(__value)},
                __is_parallel);
        },
        [&]() {
            return dpstd::__internal::__pattern_walk1_n(
                std::forward<_ExecutionPolicy>(__exec), __first, __n,
                dpstd::__internal::__op_uninitialized_fill<_Tp, _DecayedExecutionPolicy>{__value}, __is_vector,
                __is_parallel);
        });
}

// [specialized.destroy]

template <class _ExecutionPolicy, class _ForwardIterator>
dpstd::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
destroy(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    typedef typename iterator_traits<_ForwardIterator>::reference _ReferenceType;
    typedef typename std::decay<_ExecutionPolicy>::type _DecayedExecutionPolicy;

    const auto __is_parallel =
        dpstd::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector =
        dpstd::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    dpstd::__internal::__invoke_if_not(std::is_trivially_destructible<_ValueType>(), [&]() {
        dpstd::__internal::__pattern_walk1(std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                           [](_ReferenceType __val) { __val.~_ValueType(); }, __is_vector,
                                           __is_parallel);
    });
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
dpstd::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
destroy_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n)
{
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    typedef typename iterator_traits<_ForwardIterator>::reference _ReferenceType;
    typedef typename std::decay<_ExecutionPolicy>::type _DecayedExecutionPolicy;

    const auto __is_parallel =
        dpstd::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector =
        dpstd::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    return dpstd::__internal::__invoke_if_else(
        std::is_trivially_destructible<_ValueType>(), [&]() { return dpstd::__internal::__pstl_next(__first, __n); },
        [&]() {
            return dpstd::__internal::__pattern_walk1_n(std::forward<_ExecutionPolicy>(__exec), __first, __n,
                                                        [](_ReferenceType __val) { __val.~_ValueType(); }, __is_vector,
                                                        __is_parallel);
        });
}

// [uninitialized.construct.default]

template <class _ExecutionPolicy, class _ForwardIterator>
dpstd::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_default_construct(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    typedef typename iterator_traits<_ForwardIterator>::reference _ReferenceType;
    typedef typename std::decay<_ExecutionPolicy>::type _DecayedExecutionPolicy;

    const auto __is_parallel =
        dpstd::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector =
        dpstd::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    dpstd::__internal::__invoke_if_not(std::is_trivial<_ValueType>(), [&]() {
        dpstd::__internal::__pattern_walk1(
            std::forward<_ExecutionPolicy>(__exec), __first, __last,
            dpstd::__internal::__op_uninitialized_default_construct<_DecayedExecutionPolicy>{}, __is_vector,
            __is_parallel);
    });
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
dpstd::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_default_construct_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n)
{
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    typedef typename iterator_traits<_ForwardIterator>::reference _ReferenceType;
    typedef typename std::decay<_ExecutionPolicy>::type _DecayedExecutionPolicy;

    const auto __is_parallel =
        dpstd::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector =
        dpstd::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    return dpstd::__internal::__invoke_if_else(
        std::is_trivial<_ValueType>(), [&]() { return dpstd::__internal::__pstl_next(__first, __n); },
        [&]() {
            return dpstd::__internal::__pattern_walk1_n(
                std::forward<_ExecutionPolicy>(__exec), __first, __n,
                dpstd::__internal::__op_uninitialized_default_construct<_DecayedExecutionPolicy>{}, __is_vector,
                __is_parallel);
        });
}

// [uninitialized.construct.value]

template <class _ExecutionPolicy, class _ForwardIterator>
dpstd::__internal::__enable_if_execution_policy<_ExecutionPolicy, void>
uninitialized_value_construct(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    typedef typename iterator_traits<_ForwardIterator>::reference _ReferenceType;
    typedef typename std::decay<_ExecutionPolicy>::type _DecayedExecutionPolicy;

    const auto __is_parallel =
        dpstd::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector =
        dpstd::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    dpstd::__internal::__invoke_if_else(
        std::is_trivial<_ValueType>(),
        [&]() {
            dpstd::__internal::__pattern_walk_brick(
                std::forward<_ExecutionPolicy>(__exec), __first, __last,
                dpstd::__internal::__brick_fill<_ValueType, _DecayedExecutionPolicy>{_ValueType()}, __is_parallel);
        },
        [&]() {
            dpstd::__internal::__pattern_walk1(
                std::forward<_ExecutionPolicy>(__exec), __first, __last,
                dpstd::__internal::__op_uninitialized_value_construct<_DecayedExecutionPolicy>{}, __is_vector,
                __is_parallel);
        });
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
dpstd::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_value_construct_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n)
{
    typedef typename iterator_traits<_ForwardIterator>::value_type _ValueType;
    typedef typename iterator_traits<_ForwardIterator>::reference _ReferenceType;
    typedef typename std::decay<_ExecutionPolicy>::type _DecayedExecutionPolicy;

    const auto __is_parallel =
        dpstd::__internal::__is_parallelization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);
    const auto __is_vector =
        dpstd::__internal::__is_vectorization_preferred<_ExecutionPolicy, _ForwardIterator>(__exec);

    return dpstd::__internal::__invoke_if_else(
        std::is_trivial<_ValueType>(),
        [&]() {
            return dpstd::__internal::__pattern_walk_brick_n(
                std::forward<_ExecutionPolicy>(__exec), __first, __n,
                dpstd::__internal::__brick_fill_n<_ValueType, _DecayedExecutionPolicy>{_ValueType()}, __is_parallel);
        },
        [&]() {
            return dpstd::__internal::__pattern_walk1_n(
                std::forward<_ExecutionPolicy>(__exec), __first, __n,
                dpstd::__internal::__op_uninitialized_value_construct<_DecayedExecutionPolicy>{}, __is_vector,
                __is_parallel);
        });
}

} // namespace std

#endif /* _PSTL_GLUE_MEMORY_IMPL_H */
