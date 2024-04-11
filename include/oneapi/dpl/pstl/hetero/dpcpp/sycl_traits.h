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

// This file contains some specialization SYCL traits for some oneDPL types.
//
// Fancy iterators and internal functors which are device copyable when their
// template arguments are also device copyable should be explicitly specialized
// as such. This is important when template argument member variables may be
// device copyable but not trivially copyable.
// Include this header before a kernel submit SYCL code.

#ifndef _ONEDPL_SYCL_TRAITS_H
#define _ONEDPL_SYCL_TRAITS_H

#if __INTEL_LLVM_COMPILER && (__INTEL_LLVM_COMPILER < 20240100)

#    define _ONEDPL_DEVICE_COPYABLE(TYPE)                                                                              \
        template <typename... Ts>                                                                                      \
        struct sycl::is_device_copyable<TYPE<Ts...>, ::std::enable_if_t<!std::is_trivially_copyable_v<TYPE<Ts...>>>>   \
            : ::std::conjunction<sycl::is_device_copyable<Ts>...>                                                      \
        {                                                                                                              \
        };

#else

#    define _ONEDPL_DEVICE_COPYABLE(TYPE)                                                                              \
        template <typename... Ts>                                                                                      \
        struct sycl::is_device_copyable<TYPE<Ts...>> : ::std::conjunction<sycl::is_device_copyable<Ts>...>             \
        {                                                                                                              \
        };

#endif

_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__not_pred)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__reorder_pred)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__equal_value_by_pred)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__equal_value)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__not_equal_value)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__transform_functor)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__transform_if_unary_functor)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__transform_if_binary_functor)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__replace_functor)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__replace_copy_functor)

namespace oneapi::dpl::__internal
{

template <typename _SourceT>
struct fill_functor;

template <typename _Generator>
struct generate_functor;

template <typename _Pred>
struct equal_predicate;

template <typename _Tp, typename _Pred>
struct __search_n_unary_predicate;

template <typename _Predicate>
struct adjacent_find_fn;

template <class _Comp>
struct __is_heap_check;

template <typename _Predicate, typename _ValueType>
struct __create_mask_unique_copy;

template <class _Tag, typename _ExecutionPolicy, typename _Tp, typename>
struct __brick_fill;

template <class _Tag, typename _ExecutionPolicy, typename _Tp, typename>
struct __brick_fill_n;

} // namespace oneapi::dpl::__internal

_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::fill_functor)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::generate_functor)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__brick_fill)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__brick_fill_n)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__search_n_unary_predicate)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__is_heap_check)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::equal_predicate)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::adjacent_find_fn)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__internal::__create_mask_unique_copy)

namespace oneapi::dpl::__par_backend_hetero
{

template <typename _ExecutionPolicy, typename _Pred>
struct __early_exit_find_or;

} // namespace oneapi::dpl::__par_backend_hetero

_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::__par_backend_hetero::__early_exit_find_or);

namespace oneapi::dpl::unseq_backend
{

template <typename _ExecutionPolicy, typename _F>
struct walk_n;

template <typename _ExecutionPolicy, typename _F>
struct walk_adjacent_difference;

template <typename _ExecutionPolicy, ::std::uint8_t __iters_per_work_item, typename _Operation1, typename _Operation2,
          typename _Tp, typename _Commutative>
struct transform_reduce;

template <typename _ExecutionPolicy, typename _BinaryOperation1, typename _Tp>
struct reduce_over_group;

template <typename _ExecutionPolicy, typename _Pred>
struct single_match_pred_by_idx;

template <typename _ExecutionPolicy, typename _Pred>
struct multiple_match_pred;

template <typename _ExecutionPolicy, typename _Pred, typename _Tp, typename _Size>
struct n_elem_match_pred;

template <typename _ExecutionPolicy, typename _Pred>
struct first_match_pred;

template <typename _Pred, typename _Tp>
struct __create_mask;

template <typename _BinaryOp, typename _Assigner, typename _Inclusive, ::std::size_t N>
struct __copy_by_mask;

template <typename _BinaryOp, typename _Inclusive>
struct __partition_by_mask;

template <typename _Inclusive, typename _BinaryOp, typename _InitType>
struct __global_scan_functor;

template <typename _Inclusive, typename _ExecutionPolicy, typename _BinaryOperation, typename _UnaryOp,
          typename _WgAssigner, typename _GlobalAssigner, typename _DataAccessor, typename _InitType>
struct __scan;

template <typename _ExecutionPolicy, typename _Compare, typename _Size1, typename _Size2>
struct __brick_includes;

template <typename _ExecutionPolicy, typename _Compare, typename _Size1, typename _Size2, typename _IsOpDifference>
class __brick_set_op;

template <typename _BinaryOperator, typename _Size>
struct __brick_reduce_idx;

} // namespace oneapi::dpl::unseq_backend

_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::walk_n)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::walk_adjacent_difference)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::transform_reduce)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::reduce_over_group)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::single_match_pred_by_idx)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::multiple_match_pred)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::n_elem_match_pred)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::first_match_pred)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::__create_mask)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::__copy_by_mask)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::__partition_by_mask)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::__global_scan_functor)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::__scan)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::__brick_includes)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::__brick_set_op)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::unseq_backend::__brick_reduce_idx)

namespace oneapi::dpl::internal
{

enum class search_algorithm;

template <typename Comp, typename T, search_algorithm func>
struct custom_brick;

template <typename T, typename Predicate>
struct replace_if_fun;

template <typename T, typename Predicate, typename UnaryOperation>
class transform_if_stencil_fun;

template <typename ValueType, typename FlagType, typename BinaryOp>
struct segmented_scan_fun;

template <typename Output1, typename Output2>
class scatter_and_accumulate_fun;

template <typename ValueType, typename FlagType, typename BinaryOp>
struct scan_by_key_fun;

} // namespace oneapi::dpl::internal

_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::internal::custom_brick)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::internal::replace_if_fun)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::internal::scan_by_key_fun)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::internal::segmented_scan_fun)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::internal::scatter_and_accumulate_fun)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::internal::transform_if_stencil_fun)

_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::zip_iterator)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::transform_iterator)
_ONEDPL_DEVICE_COPYABLE(oneapi::dpl::permutation_iterator)

#undef _ONEDPL_DEVICE_COPYABLE

#endif // _ONEDPL_SYCL_TRAITS_H
