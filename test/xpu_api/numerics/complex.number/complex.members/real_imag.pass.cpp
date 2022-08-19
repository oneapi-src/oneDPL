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
#if TEST_STD_VER > 11
    constexpr dpl::complex<T> c1;
    static_assert(c1.real() == 0, "");
    static_assert(c1.imag() == 0, "");
    constexpr dpl::complex<T> c2(3);
    static_assert(c2.real() == 3, "");
    static_assert(c2.imag() == 0, "");
    constexpr dpl::complex<T> c3(3, 4);
    static_assert(c3.real() == 3, "");
    static_assert(c3.imag() == 4, "");
#endif
}

template <class T>
void
test()
{
    dpl::complex<T> c;
    assert(c.real() == 0);
    assert(c.imag() == 0);
    c.real(3.5);
    assert(c.real() == 3.5);
    assert(c.imag() == 0);
    c.imag(4.5);
    assert(c.real() == 3.5);
    assert(c.imag() == 4.5);
    c.real(-4.5);
    assert(c.real() == -4.5);
    assert(c.imag() == 4.5);
    c.imag(-5.5);
    assert(c.real() == -4.5);
    assert(c.imag() == -5.5);

    test_constexpr<T> ();
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    RUN_IF_DOUBLE_SUPPORT(test<double>())
    RUN_IF_LDOUBLE_SUPPORT(test<long double>())
    test_constexpr<int>();

  return 0;
}
