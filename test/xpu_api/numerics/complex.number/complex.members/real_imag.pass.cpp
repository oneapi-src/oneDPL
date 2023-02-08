//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// void real(T val);
// void imag(T val);

#include "support/test_complex.h"

template <class T>
void
test_constexpr()
{
    constexpr dpl::complex<T> c1;
    STD_COMPLEX_TESTS_STATIC_ASSERT(c1.real() == 0, "");
    STD_COMPLEX_TESTS_STATIC_ASSERT(c1.imag() == 0, "");
    constexpr dpl::complex<T> c2(3);
    STD_COMPLEX_TESTS_STATIC_ASSERT(c2.real() == 3, "");
    STD_COMPLEX_TESTS_STATIC_ASSERT(c2.imag() == 0, "");
    constexpr dpl::complex<T> c3(3, 4);
    STD_COMPLEX_TESTS_STATIC_ASSERT(c3.real() == 3, "");
    STD_COMPLEX_TESTS_STATIC_ASSERT(c3.imag() == 4, "");
}

template <class T>
void
test()
{
    dpl::complex<T> c;
    assert(c.real() == 0);
    assert(c.imag() == 0);
    c.real(3.5f);
    assert(c.real() == 3.5f);
    assert(c.imag() == 0);
    c.imag(4.5f);
    assert(c.real() == 3.5f);
    assert(c.imag() == 4.5f);
    c.real(-4.5f);
    assert(c.real() == -4.5f);
    assert(c.imag() == 4.5f);
    c.imag(-5.5f);
    assert(c.real() == -4.5f);
    assert(c.imag() == -5.5f);

    test_constexpr<T> ();
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
#if _PSTL_TEST_COMPLEX_NON_FLOAT_AVAILABLE
    test_constexpr<int>();
#endif // _PSTL_TEST_COMPLEX_NON_FLOAT_AVAILABLE

  return 0;
}
