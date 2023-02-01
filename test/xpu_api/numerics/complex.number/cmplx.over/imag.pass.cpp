//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<Arithmetic T>
//   T
//   imag(const T& x);

// Warning	G2904547A	'complex' is deprecated: warning STL4037:
// The effect of instantiating the template std::complex for any type other than float, double, or long double is unspecified.
// You can define _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING to suppress this warning. [-Wdeprecated-declarations]
#define _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING

#include "support/test_complex.h"

#include "../cases.h"

template <class T, int x>
void
test(typename std::enable_if<std::is_integral<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(dpl::imag(T(x))), double>::value), "");
    assert(dpl::imag(x) == 0);

    constexpr T val {x};
    STD_COMPLEX_TESTS_STATIC_ASSERT(dpl::imag(val) == 0, "");
    constexpr dpl::complex<T> t{val, val};
    STD_COMPLEX_TESTS_STATIC_ASSERT(t.imag() == x, "" );
}

template <class T, int x>
void
test(typename std::enable_if<!std::is_integral<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(dpl::imag(T(x))), T>::value), "");
    assert(dpl::imag(x) == 0);

    constexpr T val {x};
    STD_COMPLEX_TESTS_STATIC_ASSERT(dpl::imag(val) == 0, "");
    constexpr dpl::complex<T> t{val, val};
    STD_COMPLEX_TESTS_STATIC_ASSERT(t.imag() == x, "" );
}

template <class T>
void
test()
{
    test<T, 0>();
    test<T, 1>();
    test<T, 10>();
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    test<int>();
    test<unsigned>();
    test<long long>();

  return 0;
}
