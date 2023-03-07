// -*- C++ -*-
//===-- numeric_impl_hetero.h ---------------------------------------------===//
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

#ifndef _ONEDPL_NUMERIC_IMPL_HETERO_H
#define _ONEDPL_NUMERIC_IMPL_HETERO_H

#include <iterator>
#include "../parallel_backend.h"
#if _ONEDPL_BACKEND_SYCL
#    include "dpcpp/execution_sycl_defs.h"
#    include "algorithm_impl_hetero.h" // to use __pattern_walk2_brick
#    include "dpcpp/parallel_backend_sycl_utils.h"
#    include "dpcpp/unseq_backend_sycl.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//------------------------------------------------------------------------
// transform_reduce (version with two binary functions)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _Tp,
          typename _BinaryOperation1, typename _BinaryOperation2>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Tp>
__pattern_transform_reduce(_ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1,
                           _RandomAccessIterator2 __first2, _Tp __init, _BinaryOperation1 __binary_op1,
                           _BinaryOperation2 __binary_op2, /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    if (__first1 == __last1)
        return __init;

    using _Functor = unseq_backend::walk_n<_ExecutionPolicy, _BinaryOperation2>;
    using _RepackedTp = __par_backend_hetero::__repacked_tuple_t<_Tp>;

    auto __n = __last1 - __first1;
    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _RandomAccessIterator1>();
    auto __buf1 = __keep1(__first1, __last1);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _RandomAccessIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);

    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_RepackedTp, _BinaryOperation1, _Functor>(
               ::std::forward<_ExecutionPolicy>(__exec), __binary_op1, _Functor{__binary_op2},
               unseq_backend::__init_value<_RepackedTp>{__init}, // initial value
               __buf1.all_view(), __buf2.all_view())
        .get();
}

//------------------------------------------------------------------------
// transform_reduce (with unary and binary functions)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _ForwardIterator, typename _Tp, typename _BinaryOperation,
          typename _UnaryOperation>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Tp>
__pattern_transform_reduce(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
                           _BinaryOperation __binary_op, _UnaryOperation __unary_op, /*vector=*/::std::true_type,
                           /*parallel=*/::std::true_type)
{
    if (__first == __last)
        return __init;

    using _Functor = unseq_backend::walk_n<_ExecutionPolicy, _UnaryOperation>;
    using _RepackedTp = __par_backend_hetero::__repacked_tuple_t<_Tp>;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator>();
    auto __buf = __keep(__first, __last);

    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_RepackedTp, _BinaryOperation, _Functor>(
               ::std::forward<_ExecutionPolicy>(__exec), __binary_op, _Functor{__unary_op},
               unseq_backend::__init_value<_RepackedTp>{__init}, // initial value
               __buf.all_view())
        .get();
}

//------------------------------------------------------------------------
// transform_scan
//------------------------------------------------------------------------
template <typename T>
struct ExecutionPolicyWrapper;

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _UnaryOperation,
          typename _InitType, typename _BinaryOperation, typename _Inclusive>
void
__pattern_transform_scan_base_impl(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last,
                                   _Iterator2 __result, _UnaryOperation __unary_op, _InitType __init,
                                   _BinaryOperation __binary_op, _Inclusive)
{
    const auto __size = __last - __first;
    assert(__size > 0);
    const ::std::size_t __n = __last - __first;

    using _Type = typename _InitType::__value_type;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf1 = __keep1(__first, __last);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _Iterator2>();
    auto __buf2 = __keep2(__result, __result + __n);

    // Next power of 2 greater than or equal to __n
    auto __n_uniform = __n;
    if ((__n_uniform & (__n_uniform - 1)) != 0)
        __n_uniform = __dpl_bit_floor(__n) << 1;

    // Pessimistically only use half of the memory to take into account memory used by compiled kernel
    const ::std::size_t __max_slm_size =
        __exec.queue().get_device().template get_info<sycl::info::device::local_mem_size>() / 2;
    const auto __req_slm_size = sizeof(_Type) * __n_uniform;

    constexpr int __single_group_upper_limit = 16384;

    if (__n <= __single_group_upper_limit && __max_slm_size >= __req_slm_size)
    {
        constexpr bool __can_use_group_scan = unseq_backend::__has_known_identity<_BinaryOperation, _Type>::value;
        if constexpr (__can_use_group_scan)
        {
            oneapi::dpl::__par_backend_hetero::__pattern_transform_scan_single_group(
                std::forward<_ExecutionPolicy>(__exec), __buf1, __buf2, __n, __unary_op, __init, __binary_op,
                _Inclusive{})
                .wait();
        }
        else
        {
            oneapi::dpl::__par_backend_hetero::__pattern_transform_scan_multi_group(
                std::forward<_ExecutionPolicy>(__exec), __buf1, __buf2, __unary_op, __init, __binary_op, _Inclusive{})
                .wait();
        }
    }
    else
    {
        oneapi::dpl::__par_backend_hetero::__pattern_transform_scan_multi_group(
            std::forward<_ExecutionPolicy>(__exec), __buf1, __buf2, __unary_op, __init, __binary_op, _Inclusive{})
            .wait();
    }
}

template <typename _Iterator1, typename _Iterator2>
constexpr bool
__check_equal_iterators(_Iterator1 __it1, _Iterator2 __it2)
{
    // In-place exclusive scan works correctly only if an input and an output iterators are the same type.
    // Otherwise, there is no way to check an in-place case and a workaround below is not applied.
    if constexpr (::std::is_same_v<::std::decay_t<_Iterator1>, ::std::decay_t<_Iterator2>>)
    {
        return __it1 == __it2;
    }

    return false;
}

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _UnaryOperation,
          typename _InitType, typename _BinaryOperation, typename _Inclusive>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator2>
__pattern_transform_scan_base(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result,
                              _UnaryOperation __unary_op, _InitType __init, _BinaryOperation __binary_op, _Inclusive)
{
    if (__first == __last)
        return __result;

    const auto __n = __last - __first;

    // This is a temporary workaround for an in-place exclusive scan while the SYCL backend scan pattern is not fixed.
    const bool __is_scan_inplace_exclusive = __n > 1 && !_Inclusive{} && __check_equal_iterators(__first, __result);
    if (!__is_scan_inplace_exclusive)
    {
        __pattern_transform_scan_base_impl(__exec, __first, __last, __result, __unary_op, __init, __binary_op,
                                           _Inclusive{});
    }
    else
    {
        assert(__n > 1);
        assert(!_Inclusive{});
        assert(__check_equal_iterators(__first, __result));

        using _Type = typename _InitType::__value_type;

        auto __policy =
            __par_backend_hetero::make_wrapped_policy<ExecutionPolicyWrapper>(::std::forward<_ExecutionPolicy>(__exec));
        using _NewExecutionPolicy = decltype(__policy);

        // Create temporary buffer
        oneapi::dpl::__par_backend_hetero::__internal::__buffer<_NewExecutionPolicy, _Type> __tmp_buf(__policy, __n);
        auto __first_tmp = __tmp_buf.get();
        auto __last_tmp = __first_tmp + __n;

        // Run main algorithm and save data into temporary buffer
        __pattern_transform_scan_base_impl(__policy, __first, __last, __first_tmp, __unary_op, __init, __binary_op,
                                           _Inclusive{});

        // Move data from temporary buffer into results
        oneapi::dpl::__internal::__pattern_walk2_brick(
            ::std::forward<_NewExecutionPolicy>(__policy), __first_tmp, __last_tmp, __result,
            oneapi::dpl::__internal::__brick_move<_NewExecutionPolicy>{}, ::std::true_type{});

        //TODO: optimize copy back depending on Iterator, i.e. set_final_data for host iterator/pointer
    }

    return __result + __n;
}

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _UnaryOperation, typename _Type,
          typename _BinaryOperation, typename _Inclusive>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator2>
__pattern_transform_scan(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result,
                         _UnaryOperation __unary_op, _Type __init, _BinaryOperation __binary_op, _Inclusive,
                         /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    using _RepackedType = __par_backend_hetero::__repacked_tuple_t<_Type>;
    using _InitType = unseq_backend::__init_value<_RepackedType>;

    return __pattern_transform_scan_base(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
                                         __unary_op, _InitType{__init}, __binary_op, _Inclusive{});
}

// scan without initial element
template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _UnaryOperation,
          typename _BinaryOperation, typename _Inclusive>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator2>
__pattern_transform_scan(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result,
                         _UnaryOperation __unary_op, _BinaryOperation __binary_op, _Inclusive,
                         /*vector=*/::std::true_type, /*parallel=*/::std::true_type)
{
    using _Type = typename ::std::iterator_traits<_Iterator1>::value_type;
    using _RepackedType = __par_backend_hetero::__repacked_tuple_t<_Type>;
    using _InitType = unseq_backend::__no_init_value<_RepackedType>;

    return __pattern_transform_scan_base(::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
                                         __unary_op, _InitType{}, __binary_op, _Inclusive{});
}

//------------------------------------------------------------------------
// adjacent_difference
//------------------------------------------------------------------------

// a wrapper for the policy is required to avoid the kernel naming issue
template <typename Name>
struct adjacent_difference_wrapper
{
};

template <typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _BinaryOperation>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _ForwardIterator2>
__pattern_adjacent_difference(_ExecutionPolicy&& __exec, _ForwardIterator1 __first, _ForwardIterator1 __last,
                              _ForwardIterator2 __d_first, _BinaryOperation __op, /*vector*/ ::std::true_type,
                              /*parallel*/ ::std::true_type)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return __d_first;

    using _It1ValueT = typename ::std::iterator_traits<_ForwardIterator1>::value_type;
    using _It2ValueTRef = typename ::std::iterator_traits<_ForwardIterator2>::reference;

    _ForwardIterator2 __d_last = __d_first + __n;

#if !__SYCL_UNNAMED_LAMBDA__
    // if we have the only element, just copy it according to the specification
    if (__n == 1)
    {
        return __internal::__except_handler([&__exec, __first, __last, __d_first, __d_last, &__op]() {
            auto __wrapped_policy = __par_backend_hetero::make_wrapped_policy<adjacent_difference_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec));

            __internal::__pattern_walk2_brick(__wrapped_policy, __first, __last, __d_first,
                                              __internal::__brick_copy<decltype(__wrapped_policy)>{},
                                              ::std::true_type{});

            return __d_last;
        });
    }
    else
#endif
    {
        return __internal::__except_handler([&__exec, __first, __last, __d_first, __d_last, &__op, __n]() {
            auto __fn = [__op](_It1ValueT __in1, _It1ValueT __in2, _It2ValueTRef __out1) {
                __out1 = __op(__in2, __in1); // This move assignment is allowed by the C++ standard draft N4810
            };

            auto __keep1 =
                oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator1>();
            auto __buf1 = __keep1(__first, __last);
            auto __keep2 =
                oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _ForwardIterator2>();
            auto __buf2 = __keep2(__d_first, __d_last);

            using _Function = unseq_backend::walk_adjacent_difference<_ExecutionPolicy, decltype(__fn)>;

            oneapi::dpl::__par_backend_hetero::__parallel_for(__exec, _Function{__fn}, __n, __buf1.all_view(),
                                                              __buf2.all_view())
                .wait();

            return __d_last;
        });
    }
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_NUMERIC_IMPL_HETERO_H
