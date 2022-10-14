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

    using _Policy = _ExecutionPolicy;
    using _Functor = unseq_backend::walk_n<_Policy, _BinaryOperation2>;
    using _RepackedTp = __par_backend_hetero::__repacked_tuple_t<_Tp>;
    using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    auto __n = __last1 - __first1;
    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _RandomAccessIterator1>();
    auto __buf1 = __keep1(__first1, __last1);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _RandomAccessIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);

    auto __res =
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_RepackedTp>(
            ::std::forward<_ExecutionPolicy>(__exec),
            unseq_backend::transform_init<_Policy, _BinaryOperation1, _Functor>{__binary_op1,
                                                                                _Functor{__binary_op2}}, // transform
            unseq_backend::transform_init<_Policy, _BinaryOperation1, _NoOpFunctor>{__binary_op1, _NoOpFunctor{}},
            unseq_backend::reduce<_Policy, _BinaryOperation1, _RepackedTp>{__binary_op1}, // reduce
            unseq_backend::__init_value<_RepackedTp>{__init},                             //initial value
            __buf1.all_view(), __buf2.all_view())
            .get();

    return __res;
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

    using _Policy = _ExecutionPolicy;
    using _Functor = unseq_backend::walk_n<_Policy, _UnaryOperation>;
    using _RepackedTp = __par_backend_hetero::__repacked_tuple_t<_Tp>;
    using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator>();
    auto __buf = __keep(__first, __last);
    auto __res =
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_RepackedTp>(
            ::std::forward<_ExecutionPolicy>(__exec),
            unseq_backend::transform_init<_Policy, _BinaryOperation, _Functor>{__binary_op,
                                                                               _Functor{__unary_op}}, // transform
            unseq_backend::transform_init<_Policy, _BinaryOperation, _NoOpFunctor>{__binary_op, _NoOpFunctor{}},
            unseq_backend::reduce<_Policy, _BinaryOperation, _RepackedTp>{__binary_op}, // reduce
            unseq_backend::__init_value<_RepackedTp>{__init},                           //initial value
            __buf.all_view())
            .get();

    return __res;
}

//------------------------------------------------------------------------
// transform_scan
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _UnaryOperation,
          typename _InitType, typename _BinaryOperation, typename _Inclusive>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Iterator2>
__pattern_transform_scan_base(_ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last, _Iterator2 __result,
                              _UnaryOperation __unary_op, _InitType __init, _BinaryOperation __binary_op, _Inclusive)
{
    if (__first == __last)
        return __result;

    using _Type = typename _InitType::__value_type;
    using _Assigner = unseq_backend::__scan_assigner;
    using _NoAssign = unseq_backend::__scan_no_assign;
    using _UnaryFunctor = unseq_backend::walk_n<_ExecutionPolicy, _UnaryOperation>;
    using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    _Assigner __assign_op;
    _NoAssign __no_assign_op;
    _NoOpFunctor __get_data_op;

    auto __n = __last - __first;
    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf1 = __keep1(__first, __last);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _Iterator2>();
    auto __buf2 = __keep2(__result, __result + __n);

    oneapi::dpl::__par_backend_hetero::__parallel_transform_scan(
        ::std::forward<_ExecutionPolicy>(__exec), __buf1.all_view(), __buf2.all_view(), __binary_op, __init,
        // local scan
        unseq_backend::__scan<_Inclusive, _ExecutionPolicy, _BinaryOperation, _UnaryFunctor, _Assigner, _Assigner,
                              _NoOpFunctor, _InitType>{__binary_op, _UnaryFunctor{__unary_op}, __assign_op, __assign_op,
                                                       __get_data_op},
        // scan between groups
        unseq_backend::__scan</*inclusive=*/::std::true_type, _ExecutionPolicy, _BinaryOperation, _NoOpFunctor,
                              _NoAssign, _Assigner, _NoOpFunctor, unseq_backend::__no_init_value<_Type>>{
            __binary_op, _NoOpFunctor{}, __no_assign_op, __assign_op, __get_data_op},
        // global scan
        unseq_backend::__global_scan_functor<_Inclusive, _BinaryOperation, _InitType>{__binary_op, __init})
        .wait();

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

#endif /* _ONEDPL_NUMERIC_IMPL_HETERO_H */
