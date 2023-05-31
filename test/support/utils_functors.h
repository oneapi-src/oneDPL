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

#ifndef _UTILS_FUNCTORS_H
#define _UTILS_FUNCTORS_H

#include <type_traits>
#include <complex>

// An arbitrary binary predicate to simulate a predicate the user providing
// a custom predicate.
template <typename _Tp>
struct UserBinaryPredicate
{
    bool
    operator()(const _Tp& __x, const _Tp& __y) const
    {
        using KeyT = ::std::decay_t<_Tp>;
        return __y != KeyT(1);
    }
};
 
template <typename _Tp>
struct MaxFunctor
{
    _Tp
    operator()(const _Tp& __x, const _Tp& __y) const
    {
        return (__x < __y) ? __y : __x;
    }
};

// TODO: Investigate why we cannot call ::std::abs on complex
// types with the CUDA backend.
template <typename _Tp>
struct MaxFunctor<::std::complex<_Tp>>
{
    auto
    complex_abs(const ::std::complex<_Tp>& __x) const
    {
        return ::std::sqrt(__x.real() * __x.real() + __x.imag() * __x.imag());
    }
    ::std::complex<_Tp>
    operator()(const ::std::complex<_Tp>& __x, const ::std::complex<_Tp>& __y) const
    {
        return (complex_abs(__x) < complex_abs(__y)) ? __y : __x;
    }
};

#endif

