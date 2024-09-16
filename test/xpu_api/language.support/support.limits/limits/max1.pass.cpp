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

// max()

#include "support/test_complex.h"
#include "support/test_macros.h"

#include <oneapi/dpl/limits>
#include <climits>
#include <cfloat>
#include <cassert>

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
#   include <cwchar>
#endif

template <class T>
void
test(T expected)
{
    assert(dpl::numeric_limits<T>::max() == expected);
    assert(dpl::numeric_limits<T>::is_bounded);
    assert(dpl::numeric_limits<const T>::max() == expected);
    assert(dpl::numeric_limits<const T>::is_bounded);
    assert(dpl::numeric_limits<volatile T>::max() == expected);
    assert(dpl::numeric_limits<volatile T>::is_bounded);
    assert(dpl::numeric_limits<const volatile T>::max() == expected);
    assert(dpl::numeric_limits<const volatile T>::is_bounded);
}

ONEDPL_TEST_NUM_MAIN
{
    test<bool>(true);
    test<char>(CHAR_MAX);
    test<signed char>(SCHAR_MAX);
    test<unsigned char>(UCHAR_MAX);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<wchar_t>(WCHAR_MAX);
#endif
#ifndef TEST_HAS_NO_CHAR8_T
    test<char8_t>(UCHAR_MAX); // ??
#endif
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<char16_t>(USHRT_MAX);
    test<char32_t>(UINT_MAX);
#endif
    test<short>(SHRT_MAX);
    test<unsigned short>(USHRT_MAX);
    test<int>(INT_MAX);
    test<unsigned int>(UINT_MAX);
    test<long>(LONG_MAX);
    test<unsigned long>(ULONG_MAX);
    test<long long>(LLONG_MAX);
    test<unsigned long long>(ULLONG_MAX);
#ifndef TEST_HAS_NO_INT128
    test<__int128_t>(__int128_t(__uint128_t(-1)/2));
    test<__uint128_t>(__uint128_t(-1));
#endif
    test<float>(FLT_MAX);
    IF_DOUBLE_SUPPORT(test<double>(DBL_MAX))
    IF_LONG_DOUBLE_SUPPORT(test<long double>(LDBL_MAX))

  return 0;
}
