//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator=(const complex&);
// template<class X> complex& operator= (const complex<X>&);

#include "support/test_complex.h"

template <class T, class X>
void
test()
{
    dpl::complex<T> c;
    assert(c.real() == 0);
    assert(c.imag() == 0);
    dpl::complex<T> c2(1.5f, 2.5f);
    c = c2;
    assert(c.real() == 1.5f);
    assert(c.imag() == 2.5f);
    dpl::complex<X> c3(3.5f, -4.5f);
    c = c3;
    assert(c.real() == 3.5f);
    assert(c.imag() == -4.5f);
}

ONEDPL_TEST_NUM_MAIN
{
    test<float, float>();
    IF_DOUBLE_SUPPORT(test<float, double>())
    IF_LONG_DOUBLE_SUPPORT(test<float, long double>())

    IF_DOUBLE_SUPPORT(test<double, float>())
    IF_DOUBLE_SUPPORT(test<double, double>())
    IF_LONG_DOUBLE_SUPPORT(test<double, long double>())

    IF_LONG_DOUBLE_SUPPORT(test<long double, float>())
    IF_LONG_DOUBLE_SUPPORT(test<long double, double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double, long double>())

  return 0;
}
