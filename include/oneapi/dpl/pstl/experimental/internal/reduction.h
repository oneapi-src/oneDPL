// -*- C++ -*-
//===-- reduction.h -------------------------------------------------------===//
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

#ifndef _ONEDPL_EXPERIMENTAL_REDUCTION_H
#define _ONEDPL_EXPERIMENTAL_REDUCTION_H

#include <type_traits>
#include <functional>

#include "../../utils.h"
#include "reduction_impl.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{
inline namespace parallelism_v2
{

// Reduction functions definitions

// Generic version of ::std::reduction, all the specific ones are implemented in terms of it.
template <typename _Tp, typename _BinaryOperation>
oneapi::dpl::__internal::__reduction_object<_Tp, _BinaryOperation>
reduction(_Tp& __var, const _Tp& __identity, _BinaryOperation __combiner)
{
    return oneapi::dpl::__internal::__reduction_object<_Tp, _BinaryOperation>(__var, __identity, __combiner);
}

template <typename _Tp>
oneapi::dpl::__internal::__reduction_object<_Tp, ::std::plus<_Tp>>
reduction_plus(_Tp& __var)
{
    return oneapi::dpl::experimental::parallelism_v2::reduction(__var, _Tp(), ::std::plus<_Tp>());
}

template <typename _Tp>
oneapi::dpl::__internal::__reduction_object<_Tp, ::std::multiplies<_Tp>>
reduction_multiplies(_Tp& __var)
{
    return oneapi::dpl::experimental::parallelism_v2::reduction(__var, _Tp(1), ::std::multiplies<_Tp>());
}

template <typename _Tp>
oneapi::dpl::__internal::__reduction_object<_Tp, decltype(::std::bit_and<_Tp>{})>
reduction_bit_and(_Tp& __var)
{
    return oneapi::dpl::experimental::parallelism_v2::reduction(__var, ~_Tp(), ::std::bit_and<_Tp>{});
}

template <typename _Tp>
oneapi::dpl::__internal::__reduction_object<_Tp, decltype(::std::bit_or<_Tp>{})>
reduction_bit_or(_Tp& __var)
{
    return oneapi::dpl::experimental::parallelism_v2::reduction(__var, _Tp(), ::std::bit_or<_Tp>{});
}

template <typename _Tp>
oneapi::dpl::__internal::__reduction_object<_Tp, decltype(::std::bit_xor<_Tp>{})>
reduction_bit_xor(_Tp& __var)
{
    return oneapi::dpl::experimental::parallelism_v2::reduction(__var, _Tp(), ::std::bit_xor<_Tp>{});
}

template <typename _Tp>
oneapi::dpl::__internal::__reduction_object<_Tp, decltype(oneapi::dpl::__internal::__pstl_min{})>
reduction_min(_Tp& __var)
{
    return oneapi::dpl::experimental::parallelism_v2::reduction(__var, __var, oneapi::dpl::__internal::__pstl_min{});
}

template <typename _Tp>
oneapi::dpl::__internal::__reduction_object<_Tp, decltype(oneapi::dpl::__internal::__pstl_max{})>
reduction_max(_Tp& __var)
{
    return oneapi::dpl::experimental::parallelism_v2::reduction(__var, __var, oneapi::dpl::__internal::__pstl_max{});
}

} // namespace parallelism_v2
} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_EXPERIMENTAL_REDUCTION_H
