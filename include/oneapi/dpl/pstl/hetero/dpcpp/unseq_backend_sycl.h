// -*- C++ -*-
//===-- unseq_backend_sycl.h ----------------------------------------------===//
//
// Copyright (C) 2019-2020 Intel Corporation
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

//!!! NOTE: This file should be included under the macro _PSTL_BACKEND_SYCL
#ifndef _PSTL_unseq_backend_sycl_H
#define _PSTL_unseq_backend_sycl_H

#include <type_traits>

#include "pstl_sycl_config.h"
#include "../../utils.h"

#include <CL/sycl.hpp>

namespace oneapi
{
namespace dpl
{
namespace unseq_backend
{

namespace sycl = cl::sycl;

// helpers to encapsulate void and other types
template <typename _Tp>
using void_type = typename std::enable_if<std::is_void<_Tp>::value, _Tp>::type;
template <typename _Tp>
using non_void_type = typename std::enable_if<!std::is_void<_Tp>::value, _Tp>::type;

// a way to get value_type from both accessors and USM that is needed for transform_init
template <typename _Unknown>
struct __accessor_traits
{
};

template <typename _T, int _Dim, sycl::access::mode _AccMode, sycl::access::target _AccTarget,
          sycl::access::placeholder _Placeholder>
struct __accessor_traits<sycl::accessor<_T, _Dim, _AccMode, _AccTarget, _Placeholder>>
{
    using value_type = typename sycl::accessor<_T, _Dim, _AccMode, _AccTarget, _Placeholder>::value_type;
};

template <typename _RawArrayValueType>
struct __accessor_traits<_RawArrayValueType*>
{
    using value_type = _RawArrayValueType;
};

template <typename _ExecutionPolicy, typename _F>
struct walk_n
{
    _F __f;

    template <typename _ItemId, typename... _Ranges>
    auto
    operator()(const _ItemId __idx, _Ranges&&... __rngs) -> decltype(__f(__rngs[__idx]...))
    {
        return __f(__rngs[__idx]...);
    }
};

// If read accessor returns temporary value then __no_op returns lvalue reference to it.
// After temporary value destroying it will be a reference on invalid object.
// So let's don't call functor in case of __no_op
template <typename _ExecutionPolicy>
struct walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>
{
    oneapi::dpl::__internal::__no_op __f;

    template <typename _ItemId, typename _Range>
    auto
    operator()(const _ItemId __idx, _Range&& __rng) -> decltype(__rng[__idx])
    {
        return __rng[__idx];
    }
};

using std::get;
template <typename _ExecutionPolicy, typename _F>
struct walk2
{
    _F __f;

    template <typename _ItemId, typename _Acc>
    auto
    operator()(const _ItemId __idx, const _Acc& __inout_acc)
        -> decltype(__f(get<0>((__inout_acc)[__idx]), get<1>((__inout_acc)[__idx])))
    {
        return __f(get<0>((__inout_acc)[__idx]), get<1>((__inout_acc)[__idx]));
    }
};

//------------------------------------------------------------------------
// walk_adjacent_difference
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _F>
struct walk_adjacent_difference
{
    _F __f;

    template <typename _ItemId, typename _Acc1, typename _Acc2>
    void
    operator()(const _ItemId __idx, const _Acc1& _acc_src, _Acc2& _acc_dst)
    {
        using std::get;

        // just copy an element if it is the first one
        if (__idx == 0)
            _acc_dst[__idx] = _acc_src[__idx];
        else
            __f(_acc_src[__idx + (-1)], _acc_src[__idx], _acc_dst[__idx]);
    }
};

//------------------------------------------------------------------------
// transform_reduce
//------------------------------------------------------------------------

// calculate shift where we should start processing on current item
template <typename _NDItemId, typename _GlobalIdx, typename _SizeNIter, typename _SizeN>
_SizeN
calc_shift(const _NDItemId __item_id, const _GlobalIdx __global_idx, _SizeNIter& __n_iter, const _SizeN __n)
{
    auto __global_range_size = __item_id.get_global_range().size();

    auto __start = __n_iter * __global_idx;
    auto __global_shift = __global_idx + __n_iter * __global_range_size;
    if (__n_iter > 0 && __global_shift > __n)
    {
        __start += __n % __global_range_size - __global_idx;
    }
    else if (__global_shift < __n)
    {
        __n_iter++;
    }
    return __start;
}

template <typename _ExecutionPolicy, typename _Operation1, typename _Operation2>
struct transform_init
{
    _Operation1 __binary_op;
    _Operation2 __unary_op;

    template <typename _NDItemId, typename _Acc, typename _GlobalIdx, typename _Size, typename _AccLocal>
    void
    operator()(const _NDItemId __item_id, const _GlobalIdx __global_idx, const _Acc& __acc, _Size __n,
               const _AccLocal& __local_mem)
    {
        auto __local_idx = __item_id.get_local_id(0);
        auto __global_range_size = __item_id.get_global_range().size();
        auto __n_iter = __n / __global_range_size;
        auto __start = calc_shift(__item_id, __global_idx, __n_iter, __n);
        auto __shifted_global_idx = __global_idx + __start;

        typename __accessor_traits<_AccLocal>::value_type __res;
        if (__global_idx < __n)
        {
            __res = __unary_op(__shifted_global_idx, __acc);
        }
        // Add neighbour to the current __local_mem
        for (decltype(__n_iter) __i = 1; __i < __n_iter; ++__i)
        {
            __res = __binary_op(__res, __unary_op(__shifted_global_idx + __i, __acc));
        }
        if (__global_idx < __n)
        {
            __local_mem[__local_idx] = __res;
        }
    }
};

// write data from local memory to global
template <typename _Inclusive, typename _NDItemId, typename _GlobalIdx, typename _Size, typename _AccLocal,
          typename _InAcc, typename _OutAcc, typename _Tp, typename _Fp, typename _BinaryOp, typename _UnaryOp>
void
write_to_global(const _NDItemId __item_id, const _GlobalIdx __global_idx, const _Size __n, const _AccLocal& __local_mem,
                const _InAcc& __input, const _OutAcc& __result, _Tp __init, _Fp __f, _BinaryOp __bin_op,
                _UnaryOp __unary_op)
{
    auto __local_idx = __item_id.get_local_id(0);
    auto __group_size = __item_id.get_local_range().size();
    auto __global_range_size = __item_id.get_global_range().size();
    auto __n_iter = __n / __global_range_size;
    auto __start = calc_shift(__item_id, __global_idx, __n_iter, __n);
    auto __shifted_global_idx = __global_idx + __start;

    _Tp __shift_for_true = __init;
    if (__local_idx != 0)
        __shift_for_true = __local_mem[__local_idx - 1];
    _Tp __shift_for_false = __shifted_global_idx - __shift_for_true;

    // TODO: it needs to be refactored due to a new implementation of scan
    // inclusive scan branch
    if (_Inclusive())
    {
        for (decltype(__n_iter) __i = 0; __i < __n_iter; ++__i)
        {
            auto __unary_op__result = __unary_op(__shifted_global_idx + __i, __input);
            __shift_for_true = __bin_op(__shift_for_true, __unary_op__result);
            __shift_for_false = __bin_op(__shift_for_false, 1 - __unary_op__result);

            __f(__shift_for_true, __shift_for_false, __shifted_global_idx + __i, __input, __result);
        }
    }
    // exclusive scan branch
    else
    {
        for (decltype(__n_iter) __i = 0; __i < __n_iter; ++__i)
        {
            __f(__shift_for_true, __shift_for_false, __shifted_global_idx + __i, __input, __result);

            auto __unary_op_result = __unary_op(__shifted_global_idx + __i, __input);
            __shift_for_true = __bin_op(__shift_for_true, __unary_op_result);
            __shift_for_false = __bin_op(__shift_for_false, 1 - __unary_op_result);
        }
    }
}

// Reduce on local memory
template <typename _ExecutionPolicy, typename _BinaryOperation1, typename _Tp>
struct reduce
{
    _BinaryOperation1 __bin_op1;

    template <typename _NDItemId, typename _GlobalIdx, typename _Size, typename _AccLocal>
    _Tp
    operator()(const _NDItemId __item_id, const _GlobalIdx __global_idx, const _Size __n, const _AccLocal& __local_mem)
    {
        auto __local_idx = __item_id.get_local_id(0);
        auto __group_size = __item_id.get_local_range().size();

        auto __k = 1;
        do
        {
            __item_id.barrier(sycl::access::fence_space::local_space);
            if (__local_idx % (2 * __k) == 0 && __local_idx + __k < __group_size && __global_idx < __n &&
                __global_idx + __k < __n)
            {
                __local_mem[__local_idx] = __bin_op1(__local_mem[__local_idx], __local_mem[__local_idx + __k]);
            }
            __k *= 2;
        } while (__k < __group_size);
        return __local_mem[__local_idx];
    }
};

#if _USE_SUB_GROUPS

template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, void>::type>
using enable_if_arithmetic = T;

// Reduce on local memory with subgroups
template <typename _ExecutionPolicy, typename _Tp>
struct reduce<_ExecutionPolicy, std::plus<_Tp>, enable_if_arithmetic<_Tp>>
{
    std::plus<_Tp> __reduce;

    template <typename _NDItem, typename _GlobalIdx, typename _GlobalSize, typename _LocalAcc>
    _Tp
    operator()(_NDItem __item, _GlobalIdx __global_id, _GlobalSize __n, _LocalAcc __local_mem) const
    {
        auto __local_id = __item.get_local_id(0);
        if (__global_id >= __n)
        {
            // Fill the rest of local buffer with 0s so each of inclusive_scan method could correctly work
            // for each work-item in sub-group
            __local_mem[__local_id] = 0;
        }
        __item.barrier(sycl::access::fence_space::local_space);
        return sycl::intel::reduce(__item.get_group(), __local_mem[__local_id], sycl::intel::plus<_Tp>());
    }
};
#endif

// Matchers for early_exit_or and early_exit_find

template <typename _ExecutionPolicy, typename _Pred>
struct single_match_pred_by_idx
{
    _Pred __pred;

    template <typename _Idx, typename _Acc1, typename _Size1, typename _Acc2, typename _Size2>
    bool
    operator()(const _Idx __shifted_idx, _Acc1& __acc, const _Size1 __n, const _Acc2&, const _Size2)
    {
        return __pred(__shifted_idx, __acc);
    }
};

template <typename _ExecutionPolicy, typename _Pred>
struct single_match_pred : single_match_pred_by_idx<_ExecutionPolicy, walk_n<_ExecutionPolicy, _Pred>>
{
    single_match_pred(_Pred __p) : single_match_pred_by_idx<_ExecutionPolicy, walk_n<_ExecutionPolicy, _Pred>>{__p} {}
};

template <typename _ExecutionPolicy, typename _Pred>
struct multiple_match_pred
{
    _Pred __pred;

    template <typename _Idx, typename _Acc1, typename _Size1, typename _Acc2, typename _Size2>
    bool
    operator()(const _Idx __shifted_idx, _Acc1& __acc, const _Size1 __n, const _Acc2& __s_acc, const _Size2 __s_n)
    {
        // if __shifted_idx > __n - __s_n then subrange bigger than original range.
        // So the second range is not a subrange of the first range
        bool __result = __shifted_idx <= __n - __s_n;
        const auto __total_shift = __shifted_idx;

        for (_Size2 __ii = 0; __ii < __s_n && __result; ++__ii)
            __result = __pred(__acc[__total_shift + __ii], __s_acc[__ii]);

        return __result;
    }
};

template <typename _ExecutionPolicy, typename _Pred, typename _Tp, typename _Size>
struct n_elem_match_pred
{
    _Pred __pred;
    _Tp __value;
    _Size __count;

    template <typename _Idx, typename _Acc1, typename _Size1, typename _Acc2, typename _Size2>
    bool
    operator()(const _Idx __shifted_idx, const _Acc1& __acc, const _Size1 __n, const _Acc2&, const _Size2)
    {

        bool __result = ((__shifted_idx + __count) <= __n);
        const auto __total_shift = __shifted_idx;

        for (auto __idx = 0; __idx < __count && __result; ++__idx)
            __result = __pred(__acc[__total_shift + __idx], __value);

        return __result;
    }
};

template <typename _ExecutionPolicy, typename _Pred>
struct first_match_pred
{
    _Pred __pred;

    template <typename _Idx, typename _Acc1, typename _Size1, typename _Acc2, typename _Size2>
    bool
    operator()(const _Idx __shifted_idx, const _Acc1& __acc, const _Size1 __n, const _Acc2& __s_acc, const _Size2 __s_n)
    {

        // assert: __shifted_idx < __n
        const auto __elem = __acc[__shifted_idx];

        for (auto __idx = 0; __idx < __s_n; ++__idx)
            if (__pred(__elem, __s_acc[__idx]))
                return true;

        return false;
    }
};

//------------------------------------------------------------------------
// scan
//------------------------------------------------------------------------

// mask assigner for tuples
template <std::size_t N>
struct __mask_assigner
{
    template <typename _Acc, typename _OutAcc, typename _OutIdx, typename _InAcc, typename _InIdx>
    void
    operator()(const _Acc& __acc, const _OutAcc& __out_acc, const _OutIdx __out_idx, const _InAcc& __in_acc,
               const _InIdx __in_idx)
    {
        using std::get;
        get<N>(__acc)[__out_idx] = __in_acc[__in_idx];
    }
};

// data assigners and accessors for transform_scan
struct __scan_assigner
{
    template <typename _OutAcc, typename _OutIdx, typename _InAcc, typename _InIdx>
    void
    operator()(const _OutAcc& __out_acc, const _OutIdx __out_idx, const _InAcc& __in_acc, _InIdx __in_idx)
    {
        __out_acc[__out_idx] = __in_acc[__in_idx];
    }

    template <typename _Acc, typename _OutAcc, typename _OutIdx, typename _InAcc, typename _InIdx>
    void
    operator()(const _Acc&, const _OutAcc& __out_acc, const _OutIdx __out_idx, const _InAcc& __in_acc, _InIdx __in_idx)
    {
        __out_acc[__out_idx] = __in_acc[__in_idx];
    }
};

struct __scan_no_assign
{
    template <typename _OutAcc, typename _OutIdx, typename _InAcc, typename _InIdx>
    void
    operator()(const _OutAcc& __out_acc, const _OutIdx __out_idx, const _InAcc& __in_acc, const _InIdx __in_idx)
    {
    }
};

// types of initial value for parallel_transform_scan
template <typename _InitType>
struct __scan_init
{
    _InitType __value;
    using __value_type = _InitType;
};

template <typename _InitType>
struct __scan_no_init
{
    using __value_type = _InitType;
};

// structure for the correct processing of the initial scan element
template <typename _InitType>
struct __scan_init_processing
{
    template <typename _Acc, typename _Index>
    void
    operator()(const __scan_init<_InitType>& __init, const _Acc& __out_acc, _Index __idx)
    {
        __out_acc[__idx] = __init.__value;
    }
    template <typename _Acc, typename _Index>
    void
    operator()(const __scan_no_init<_InitType>&, const _Acc&, _Index)
    {
    }

    template <typename _Acc, typename _Index, typename _BinaryOp, typename _UnaryOp>
    void
    operator()(const __scan_init<_InitType>& __init, const _Acc& __local_acc, _Index __idx, _BinaryOp __bin_op,
               _UnaryOp __unary_op)
    {
        __local_acc[__idx] = __bin_op(__unary_op(__idx, __local_acc), __init.__value);
    }
    template <typename _Acc, typename _Index, typename _BinaryOp, typename _UnaryOp>
    void
    operator()(const __scan_no_init<_InitType>&, const _Acc&, _Index, _BinaryOp, _UnaryOp)
    {
    }
};

// functors for scan
// If we work with tuples we might have a situation when internal tuple is assigned to std::tuple.
// In order for this to work, we need explicitly cast our internal tuple to std::tuple type which
// corresponds to it(i.e. internal::tuple<Ts...> -> std::tuple<Ts...>) so assignment could
// work properly. Do nothing for other types or if both operands are internal tuples.
// Here and below we use __get_tuple_type applying decay to reduce the number of specializations
// if the type is actually a reference.
template <typename _BinaryOp, typename _UnaryOp, typename _Inclusive, std::size_t N>
struct __copy_by_mask
{
    _BinaryOp __binary_op;
    _UnaryOp __unary_op;

    template <typename _Item, typename _OutAcc, typename _InAcc, typename _WgSumsAcc, typename _Size,
              typename _SizePerWg>
    void
    operator()(_Item __item, const _OutAcc& __out_acc, const _InAcc& __in_acc, const _WgSumsAcc& __wg_sums_acc,
               _Size __n, _SizePerWg __size_per_wg)
    {
        using std::get;
        auto __item_idx = __item.get_linear_id();
        if (__item_idx < __n && get<N>(__in_acc)[__item_idx])
        {
            auto __out_idx = get<N>(__in_acc)[__item_idx] - 1;
            using __tuple_type = typename __internal::__get_tuple_type<
                typename std::decay<decltype(get<0>(__in_acc)[__item_idx])>::type,
                typename std::decay<decltype(__out_acc[__out_idx])>::type>::__type;
            // calculation of position for copy
            if (__item_idx >= __size_per_wg)
            {
                auto __wg_sums_idx = __item_idx / __size_per_wg - 1;
                __out_idx = __binary_op(__out_idx, __wg_sums_acc[__wg_sums_idx]);
            }
            if (__item_idx % __size_per_wg == 0 || (get<N>(__in_acc)[__item_idx] != get<N>(__in_acc)[__item_idx - 1]))
                __out_acc[__out_idx] = static_cast<__tuple_type>(get<0>(__in_acc)[__item_idx]);
        }
    }
};

template <typename _BinaryOp, typename _UnaryOp, typename _Inclusive>
struct __partition_by_mask
{
    _BinaryOp __binary_op;
    _UnaryOp __unary_op;

    template <typename _Item, typename _OutAcc, typename _InAcc, typename _WgSumsAcc, typename _Size,
              typename _SizePerWg>
    void
    operator()(_Item __item, const _OutAcc& __out_acc, const _InAcc& __in_acc, const _WgSumsAcc& __wg_sums_acc,
               _Size __n, _SizePerWg __size_per_wg)
    {
        using std::get;
        auto __item_idx = __item.get_linear_id();
        if (__item_idx < __n)
        {
            using std::get;
            using __in_type = typename std::decay<decltype(get<0>(__in_acc)[__item_idx])>::type;
            auto __wg_sums_idx = __item_idx / __size_per_wg;
            bool __not_first_wg = __item_idx >= __size_per_wg;
            if (get<1>(__in_acc)[__item_idx] &&
                (__item_idx % __size_per_wg == 0 || get<1>(__in_acc)[__item_idx] != get<1>(__in_acc)[__item_idx - 1]))
            {
                auto __out_idx = get<1>(__in_acc)[__item_idx] - 1;
                if (__not_first_wg)
                    __out_idx = __binary_op(__out_idx, __wg_sums_acc[__wg_sums_idx - 1]);
                using __tuple_type = typename __internal::__get_tuple_type<
                    __in_type, typename std::decay<decltype(get<0>(__out_acc)[__out_idx])>::type>::__type;
                get<0>(__out_acc)[__out_idx] = static_cast<__tuple_type>(get<0>(__in_acc)[__item_idx]);
            }
            else
            {
                auto __out_idx = __item_idx - get<1>(__in_acc)[__item_idx];
                if (__not_first_wg)
                    __out_idx -= __wg_sums_acc[__wg_sums_idx - 1];
                using __tuple_type = typename __internal::__get_tuple_type<
                    __in_type, typename std::decay<decltype(get<1>(__out_acc)[__out_idx])>::type>::__type;
                get<1>(__out_acc)[__out_idx] = static_cast<__tuple_type>(get<0>(__in_acc)[__item_idx]);
            }
        }
    }
};

template <typename _BinaryOp, typename _UnaryOp, typename _Inclusive, typename _InitType>
struct __global_scan_functor
{
    _BinaryOp __binary_op;
    _UnaryOp __unary_op;

    template <typename _Item, typename _OutAcc, typename _InAcc, typename _WgSumsAcc, typename _Size,
              typename _SizePerWg>
    void
    operator()(_Item __item, const _OutAcc& __out_acc, const _InAcc& _in_acc, const _WgSumsAcc& __wg_sums_acc,
               _Size __n, _SizePerWg __size_per_wg)
    {
        // the first elements of each workgroup (from the second stage of __parallel_transform_scan) should be processed by __unary_op
        auto __item_idx = __item.get_linear_id();
        constexpr bool __inc_scan_no_init =
            std::is_same<_InitType, __scan_no_init<typename _InitType::__value_type>>::value && _Inclusive{};
        if (__inc_scan_no_init && __item_idx == 0)
            __out_acc[0] = __unary_op(0, __out_acc);

        // elements from the first workgroup are already processed
        constexpr auto __shift = _Inclusive{} ? 0 : 1;
        __item_idx += __shift;
        if (__item_idx >= __size_per_wg && __item_idx < __n)
        {
            using std::get;
            using __out_type = typename std::decay<decltype(__out_acc[__item_idx])>::type;
            auto __wg_sums_idx = (__item_idx - __shift) / __size_per_wg - 1;
            if ((__item_idx - __shift) % __size_per_wg == 0)
            {
                using __in_type = typename std::decay<decltype(
                    __binary_op(__wg_sums_acc[__wg_sums_idx], __unary_op(__item_idx, __out_acc)))>::type;
                __out_acc[__item_idx] =
                    static_cast<typename __internal::__get_tuple_type<__in_type, __out_type>::__type>(
                        __binary_op(__wg_sums_acc[__wg_sums_idx], __unary_op(__item_idx, __out_acc)));
            }
            else
            {
                using __in_type = typename std::decay<decltype(
                    __binary_op(__wg_sums_acc[__wg_sums_idx], __out_acc[__item_idx]))>::type;
                __out_acc[__item_idx] =
                    static_cast<typename __internal::__get_tuple_type<__in_type, __out_type>::__type>(
                        __binary_op(__wg_sums_acc[__wg_sums_idx], __out_acc[__item_idx]));
            }
        }
    }
};

// TODO: more performant optimization for simple types based on work group algorithms needs to be implemented
template <typename _Inclusive, typename _ExecutionPolicy, typename _BinaryOperation, typename _UnaryOp,
          typename _WgAssigner, typename _GlobalAssigner, typename _DataAccessor, typename _InitType>
struct __scan
{
    _BinaryOperation __bin_op;
    _UnaryOp __unary_op;
    _WgAssigner __wg_assigner;
    _GlobalAssigner __gl_assigner;
    _DataAccessor __data_acc;

    template <typename _NDItemId, typename _Size, typename _AccLocal, typename _InAcc, typename _OutAcc,
              typename _WGSumsAcc, typename _SizePerWG, typename _WGSize, typename _ItersPerWG>
    void operator()(_NDItemId __item, _Size __n, const _AccLocal& __local_acc, const _InAcc& __acc,
                    const _OutAcc& __out_acc, const _WGSumsAcc& __wg_sums_acc, _SizePerWG __size_per_wg,
                    _WGSize __wgroup_size, _ItersPerWG __iters_per_wg,
                    _InitType __init = __scan_no_init<typename _InitType::__value_type>{})
    {
        auto __group_id = __item.get_group(0);
        auto __global_id = __item.get_global_id(0);
        auto __local_id = __item.get_local_id(0);
        auto __use_init = __scan_init_processing<typename _InitType::__value_type>{};

        auto __shift = 0;
        __internal::__invoke_if_not(_Inclusive{}, [&]() {
            __shift = 1;
            if (__global_id == 0)
                __use_init(__init, __out_acc, __global_id);
            __item.barrier();
        });

        auto __adjusted_global_id = __local_id + __size_per_wg * __group_id;
        for (auto __iter = 0; __iter < __iters_per_wg; ++__iter, __adjusted_global_id += __wgroup_size)
        {
            if (__adjusted_global_id < __n)
                __local_acc[__local_id] = __data_acc(__adjusted_global_id, __acc);
            if (__local_id == 0 && __iter > 0)
                __local_acc[0] = __bin_op(__local_acc[__wgroup_size], __unary_op(0, __local_acc));
            if (__global_id == 0 && __iter == 0)
                __use_init(__init, __local_acc, __global_id, __bin_op, __unary_op);

            // 1. reduce
            auto __k = 1;
            do
            {
                __item.barrier(sycl::access::fence_space::local_space);
                if (__local_id % (2 * __k) == 0 && __local_id + __k < __wgroup_size && __adjusted_global_id + __k < __n)
                {
                    if (__global_id == 0 && __iter == 0 && __k == 1)
                        __internal::__invoke_if_else(
                            std::is_same<_InitType, __scan_init<typename _InitType::__value_type>>{},
                            [&] { __local_acc[1] = __bin_op(__local_acc[0], __unary_op(1, __local_acc)); },
                            [&] { __local_acc[1] = __bin_op(__unary_op(0, __local_acc), __unary_op(1, __local_acc)); });
                    else if (__local_id == 0 && __iter > 0 && __k == 1)
                        __local_acc[1] = __bin_op(__local_acc[0], __unary_op(1, __local_acc));
                    else if (__k == 1)
                        __local_acc[__local_id + 1] =
                            __bin_op(__unary_op(__local_id, __local_acc), __unary_op(__local_id + 1, __local_acc));
                    else if (__local_id + 2 * __k - 1 < __wgroup_size)
                        __local_acc[__local_id + 2 * __k - 1] =
                            __bin_op(__local_acc[__local_id + __k - 1], __local_acc[__local_id + 2 * __k - 1]);
                }
                __k *= 2;
            } while (__k < __wgroup_size);
            __item.barrier(sycl::access::fence_space::local_space);

            // 2. scan
            auto __partial_sums = __local_acc[__local_id];
            __k = 2;
            do
            {
                auto __shifted_local_id = __local_id - __local_id % __k - 1;
                if (__shifted_local_id >= 0 && __adjusted_global_id < __n && __local_id % (2 * __k) >= __k &&
                    __local_id % (2 * __k) < 2 * __k - 1)
                {
                    if ((__local_id - __k) % (2 * __k) == 0)
                        __partial_sums = __bin_op(__local_acc[__shifted_local_id], __unary_op(__local_id, __local_acc));
                    else
                        __partial_sums = __bin_op(__local_acc[__shifted_local_id], __partial_sums);
                }
                __k *= 2;
            } while (__k < __wgroup_size);
            __item.barrier(sycl::access::fence_space::local_space);
            __local_acc[__local_id] = __partial_sums;
            if (__local_id == __wgroup_size - 1)
                __local_acc[__wgroup_size] = __local_acc[__local_id];
            __item.barrier(sycl::access::fence_space::local_space);

            if (__adjusted_global_id < __n)
                __gl_assigner(__acc, __out_acc, __adjusted_global_id + __shift, __local_acc, __local_id);

            if (__adjusted_global_id == __n - 1)
                __wg_assigner(__wg_sums_acc, __group_id, __local_acc, __local_id);
        }

        if (__local_id == __wgroup_size - 1 && __adjusted_global_id - __wgroup_size < __n)
            __wg_assigner(__wg_sums_acc, __group_id, __local_acc, __local_id);
    }
};

//------------------------------------------------------------------------
// __brick_includes
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Compare, typename _Size1, typename _Size2>
struct __brick_includes
{
    _Compare __comp;
    _Size1 __na;
    _Size2 __nb;

    __brick_includes(_Compare __c, _Size1 __n1, _Size2 __n2) : __comp(__c), __na(__n1), __nb(__n2) {}

    template <typename _ItemId, typename _Acc1, typename _Acc2>
    bool
    operator()(_ItemId __idx, const _Acc1& __b_acc, _Size1, const _Acc2& __a_acc, _Size2)
    {
        using std::get;

        auto __a = __a_acc;
        auto __b = __b_acc;

        auto __a_beg = _Size1(0);
        auto __a_end = __na;

        auto __b_beg = _Size2(0);
        auto __b_end = __nb;

        // testing __comp(*__first2, *__first1) or __comp(*(__last1 - 1), *(__last2 - 1))
        if ((__idx == 0 && __comp(__b[__b_beg + 0], __a[__a_beg + 0])) ||
            (__idx == __nb - 1 && __comp(__a[__a_end - 1], __b[__b_end - 1])))
            return true; //__a doesn't include __b

        const auto __idx_b = __b_beg + __idx;
        const auto __val_b = __b[__idx_b];
        auto __res = __internal::__pstl_lower_bound(__a, __a_beg, __a_end, __val_b, __comp);

        // {a} < {b} or __val_b != __a[__res]
        if (__res == __a_end || __comp(__val_b, __a[__res]))
            return true; //__a doesn't include __b

        auto __val_a = __a[__res];

        //searching number of duplication
        const auto __count_a = __internal::__pstl_right_bound(__a, __res, __a_end, __val_a, __comp) - __res + __res -
                               __internal::__pstl_left_bound(__a, __a_beg, __res, __val_a, __comp);

        const auto __count_b = __internal::__pstl_right_bound(__b, _Size2(__idx_b), __b_end, __val_b, __comp) -
                               __idx_b + __idx_b -
                               __internal::__pstl_left_bound(__b, __b_beg, _Size2(__idx_b), __val_b, __comp);

        return __count_b > __count_a; //false means __a includes __b
    }
};

//------------------------------------------------------------------------
// reverse
//------------------------------------------------------------------------
template <typename _Size>
struct __reverse_functor
{
    _Size __size;
    template <typename _Idx, typename _Accessor>
    void
    operator()(const _Idx __idx, const _Accessor& __acc)
    {
        std::swap(__acc[__idx], __acc[__size - __idx - 1]);
    }
};

//------------------------------------------------------------------------
// reverse_copy
//------------------------------------------------------------------------
template <typename _Size>
struct __reverse_copy
{
    _Size __size;
    template <typename _Idx, typename _AccessorSrc, typename _AccessorDst>
    void
    operator()(const _Idx __idx, const _AccessorSrc& __acc1, _AccessorDst& __acc2)
    {
        __acc2[__idx] = __acc1[__size - __idx - 1];
    }
};

//------------------------------------------------------------------------
// rotate_copy
//------------------------------------------------------------------------
template <typename _Size>
struct __rotate_copy
{
    _Size __size;
    _Size __shift;
    template <typename _Idx, typename _AccessorSrc, typename _AccessorDst>
    void
    operator()(const _Idx __idx, const _AccessorSrc& __acc1, _AccessorDst& __acc2)
    {
        __acc2[__idx] = __acc1[(__shift + __idx) % __size];
    }
};

//------------------------------------------------------------------------
// brick_set_op for difference and intersection operations
//------------------------------------------------------------------------
struct _IntersectionTag : public std::false_type
{
};
struct _DifferenceTag : public std::true_type
{
};

template <typename _ExecutionPolicy, typename _Compare, typename _Size1, typename _Size2, typename _IsOpDifference>
class __brick_set_op
{
    _Compare __comp;
    _Size1 __na;
    _Size2 __nb;

  public:
    __brick_set_op(_Compare __c, _Size1 __n1, _Size2 __n2) : __comp(__c), __na(__n1), __nb(__n2) {}

    template <typename _ItemId, typename _Acc>
    bool
    operator()(_ItemId __idx, const _Acc& __inout_acc)
    {
        using std::get;
        auto __a = get<0>(__inout_acc); // first sequence
        auto __b = get<1>(__inout_acc); // second sequence
        auto __c = get<2>(__inout_acc); // mask buffer

        auto __a_beg = _Size1(0);
        auto __a_end = __na;

        auto __b_beg = _Size2(0);
        auto __b_end = __nb;

        auto __idx_c = __idx;

        const auto __idx_a = __idx;
        auto __val_a = __a[__a_beg + __idx_a];

        auto __res = __internal::__pstl_lower_bound(__b, _Size2(0), __nb, __val_a, __comp);

        bool bres = _IsOpDifference(); //initialization in true in case of difference operation; false - intersection.
        if (__res == __nb || __comp(__val_a, __b[__b_beg + __res]))
        {
            // there is no __val_a in __b, so __b in the defference {__a}/{__b};
        }
        else
        {
            auto __val_b = __b[__b_beg + __res];

            //Difference operation logic: if number of duplication in __a on left side from __idx > total number of
            //duplication in __b than a mask is 1

            //Intersection operation logic: if number of duplication in __a on left side from __idx <= total number of
            //duplication in __b than a mask is 1

            const _Size1 __count_a_left =
                __idx_a - __internal::__pstl_left_bound(__a, _Size1(0), _Size1(__idx_a), __val_a, __comp) + 1;

            const _Size2 __count_b = __internal::__pstl_right_bound(__b, _Size2(__res), __nb, __val_b, __comp) - __res +
                                     __res -
                                     __internal::__pstl_left_bound(__b, _Size2(0), _Size2(__res), __val_b, __comp);

            bres = __internal::__invoke_if_else(_IsOpDifference(),
                                                [&]() { return __count_a_left > __count_b; }, /*difference*/
                                                [&]() { return __count_a_left <= __count_b; } /*intersection*/);
        }
        __c[__idx_c] = bres; //store a mask
        return bres;
    }
};

} // namespace unseq_backend
} // namespace dpl
} // namespace oneapi

#endif /* _PSTL_unseq_backend_sycl_H */
