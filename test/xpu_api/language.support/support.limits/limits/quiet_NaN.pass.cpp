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

// test numeric_limits

// quiet_NaN()

#include "support/test_complex.h"
#include "support/test_macros.h"

#include <oneapi/dpl/limits>
#include <oneapi/dpl/cmath>
#include <type_traits>
#include <cassert>

template <class T>
void
test_imp(std::true_type)
{
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-constant-compare"
#endif

    assert(dpl::isnan(dpl::numeric_limits<T>::quiet_NaN()));
    assert(dpl::isnan(dpl::numeric_limits<const T>::quiet_NaN()));
    assert(dpl::isnan(dpl::numeric_limits<volatile T>::quiet_NaN()));
    assert(dpl::isnan(dpl::numeric_limits<const volatile T>::quiet_NaN()));

#ifdef __clang__
#pragma clang diagnostic pop
#endif
}

template <class T>
void
test_imp(std::false_type)
{
    assert(dpl::numeric_limits<T>::quiet_NaN() == T());
    assert(dpl::numeric_limits<const T>::quiet_NaN() == T());
    assert(dpl::numeric_limits<volatile T>::quiet_NaN() == T());
    assert(dpl::numeric_limits<const volatile T>::quiet_NaN() == T());
}

template <class T>
inline
void
test()
{
    test_imp<T>(std::is_floating_point<T>());
}

ONEDPL_TEST_NUM_MAIN
{
    test<bool>();
    test<char>();
    test<signed char>();
    test<unsigned char>();
    test<wchar_t>();
#ifndef TEST_HAS_NO_CHAR8_T
    test<char8_t>();
#endif
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<char16_t>();
    test<char32_t>();
#endif
    test<short>();
    test<unsigned short>();
    test<int>();
    test<unsigned int>();
    test<long>();
    test<unsigned long>();
    test<long long>();
    test<unsigned long long>();
#ifndef TEST_HAS_NO_INT128
    test<__int128_t>();
    test<__uint128_t>();
#endif
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())

  return 0;
}
