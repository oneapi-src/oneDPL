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

#ifndef _ONEDPL_numeric_ranges_impl_hetero_H
#define _ONEDPL_numeric_ranges_impl_hetero_H

#include "../numeric_fwd.h"
#include "../parallel_backend.h"

#if _ONEDPL_BACKEND_SYCL
#    include "dpcpp/utils_ranges_sycl.h"
#    include "dpcpp/unseq_backend_sycl.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{
namespace __ranges
{

//------------------------------------------------------------------------
// transform_reduce (version with two binary functions)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp, typename _BinaryOperation1,
          typename _BinaryOperation2>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Tp>
__pattern_transform_reduce(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _Tp __init,
                           _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2)
{
    if (__rng1.empty())
        return __init;

    using _Policy = _ExecutionPolicy;
    using _Functor = unseq_backend::walk_n<_Policy, _BinaryOperation2>;
    using _RepackedTp = oneapi::dpl::__par_backend_hetero::__repacked_tuple_t<_Tp>;
    using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    _RepackedTp __res =
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_RepackedTp>(
            __exec,
            unseq_backend::transform_init<_Policy, _BinaryOperation1, _Functor>{__binary_op1,
                                                                                _Functor{__binary_op2}}, // transform
            unseq_backend::transform_init<_Policy, _BinaryOperation1, _NoOpFunctor>{__binary_op1, _NoOpFunctor{}},
            unseq_backend::reduce<_Policy, _BinaryOperation1, _RepackedTp>{__binary_op1}, // reduce
            unseq_backend::__init_value<_RepackedTp>{__init},                             //initial value
            ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2))
            .get();

    return __res;
}

//------------------------------------------------------------------------
// transform_reduce (with unary and binary functions)
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range, typename _Tp, typename _BinaryOperation, typename _UnaryOperation>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy, _Tp>
__pattern_transform_reduce(_ExecutionPolicy&& __exec, _Range&& __rng, _Tp __init, _BinaryOperation __binary_op,
                           _UnaryOperation __unary_op)
{
    if (__rng.empty())
        return __init;

    using _Policy = _ExecutionPolicy;
    using _Functor = unseq_backend::walk_n<_Policy, _UnaryOperation>;
    using _RepackedTp = oneapi::dpl::__par_backend_hetero::__repacked_tuple_t<_Tp>;
    using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    _RepackedTp __res =
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_RepackedTp>(
            __exec,
            unseq_backend::transform_init<_Policy, _BinaryOperation, _Functor>{__binary_op,
                                                                               _Functor{__unary_op}}, // transform
            unseq_backend::transform_init<_Policy, _BinaryOperation, _NoOpFunctor>{__binary_op, _NoOpFunctor{}},
            unseq_backend::reduce<_Policy, _BinaryOperation, _RepackedTp>{__binary_op}, // reduce
            unseq_backend::__init_value<_RepackedTp>{__init},                           //initial value
            ::std::forward<_Range>(__rng))
            .get();

    return __res;
}

//------------------------------------------------------------------------
// transform_scan
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation, typename _InitType,
          typename _BinaryOperation, typename _Inclusive>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range2>>
__pattern_transform_scan_base(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _UnaryOperation __unary_op,
                              _InitType __init, _BinaryOperation __binary_op, _Inclusive)
{
    if (__rng1.empty())
        return 0;

    using _Type = typename _InitType::__value_type;
    using _Assigner = unseq_backend::__scan_assigner;
    using _NoAssign = unseq_backend::__scan_no_assign;
    using _UnaryFunctor = unseq_backend::walk_n<_ExecutionPolicy, _UnaryOperation>;
    using _NoOpFunctor = unseq_backend::walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>;

    _Assigner __assign_op;
    _NoAssign __no_assign_op;
    _NoOpFunctor __get_data_op;

    oneapi::dpl::__par_backend_hetero::__parallel_transform_scan(
        ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2),
        __binary_op, __init,
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
    return __rng1.size();
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation, typename _Type,
          typename _BinaryOperation, typename _Inclusive>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range2>>
__pattern_transform_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _UnaryOperation __unary_op,
                         _Type __init, _BinaryOperation __binary_op, _Inclusive)
{
    using _RepackedType = __par_backend_hetero::__repacked_tuple_t<_Type>;
    using _InitType = unseq_backend::__init_value<_RepackedType>;

    return __pattern_transform_scan_base(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                                         ::std::forward<_Range2>(__rng2), __unary_op, _InitType{__init}, __binary_op,
                                         _Inclusive{});
}

// scan without initial element
template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation,
          typename _BinaryOperation, typename _Inclusive>
oneapi::dpl::__internal::__enable_if_hetero_execution_policy<_ExecutionPolicy,
                                                             oneapi::dpl::__internal::__difference_t<_Range2>>
__pattern_transform_scan(_ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2, _UnaryOperation __unary_op,
                         _BinaryOperation __binary_op, _Inclusive)
{
    using _Type = oneapi::dpl::__internal::__value_t<_Range1>;
    using _RepackedType = __par_backend_hetero::__repacked_tuple_t<_Type>;
    using _InitType = unseq_backend::__no_init_value<_RepackedType>;

    return __pattern_transform_scan_base(::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range1>(__rng1),
                                         ::std::forward<_Range2>(__rng2), __unary_op, _InitType{}, __binary_op,
                                         _Inclusive{});
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_numeric_ranges_impl_hetero_H */
