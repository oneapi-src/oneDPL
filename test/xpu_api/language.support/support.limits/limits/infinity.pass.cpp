//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// infinity()

#include "support/test_complex.h"

#include <oneapi/dpl/limits>
#include <cfloat>
#include <cassert>

template <class T>
void
test(T expected)
{
    assert(dpl::numeric_limits<T>::infinity() == expected);
    assert(dpl::numeric_limits<const T>::infinity() == expected);
    assert(dpl::numeric_limits<volatile T>::infinity() == expected);
    assert(dpl::numeric_limits<const volatile T>::infinity() == expected);
}

extern float zero;

ONEDPL_TEST_NUM_MAIN
{
    test<bool>(false);
    test<char>(0);
    test<signed char>(0);
    test<unsigned char>(0);
    test<wchar_t>(0);
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t>(0);
#endif
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<char16_t>(0);
    test<char32_t>(0);
#endif
    test<short>(0);
    test<unsigned short>(0);
    test<int>(0);
    test<unsigned int>(0);
    test<long>(0);
    test<unsigned long>(0);
    test<long long>(0);
    test<unsigned long long>(0);
#ifndef TEST_HAS_NO_INT128
    test<__int128_t>(0);
    test<__uint128_t>(0);
#endif
    test<float>(1.f/zero);
    IF_DOUBLE_SUPPORT(test<double>(1./zero))
    IF_LONG_DOUBLE_SUPPORT(test<long double>(1./zero))

  return 0;
}

float zero = 0;
