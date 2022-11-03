//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// constexpr complex(const T& re = T(), const T& im = T());

#include "support/test_complex.h"

template <class T>
void
test()
{
    {
    const dpl::complex<T> c;
    assert(c.real() == 0);
    assert(c.imag() == 0);
    }
    {
    const dpl::complex<T> c = 7.5;
    assert(c.real() == 7.5);
    assert(c.imag() == 0);
    }
    {
    const dpl::complex<T> c(8.5);
    assert(c.real() == 8.5);
    assert(c.imag() == 0);
    }
    {
    const dpl::complex<T> c(10.5, -9.5);
    assert(c.real() == 10.5);
    assert(c.imag() == -9.5);
    }

    {
    constexpr dpl::complex<T> c;
    STD_COMPLEX_TESTS_STATIC_ASSERT(c.real() == 0, "");
    STD_COMPLEX_TESTS_STATIC_ASSERT(c.imag() == 0, "");
    }
    {
    constexpr dpl::complex<T> c = 7.5;
    STD_COMPLEX_TESTS_STATIC_ASSERT(c.real() == 7.5, "");
    STD_COMPLEX_TESTS_STATIC_ASSERT(c.imag() == 0, "");
    }
    {
    constexpr dpl::complex<T> c(8.5);
    STD_COMPLEX_TESTS_STATIC_ASSERT(c.real() == 8.5, "");
    STD_COMPLEX_TESTS_STATIC_ASSERT(c.imag() == 0, "");
    }
    {
    constexpr dpl::complex<T> c(10.5, -9.5);
    STD_COMPLEX_TESTS_STATIC_ASSERT(c.real() == 10.5, "");
    STD_COMPLEX_TESTS_STATIC_ASSERT(c.imag() == -9.5, "");
    }
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())

  return 0;
}
