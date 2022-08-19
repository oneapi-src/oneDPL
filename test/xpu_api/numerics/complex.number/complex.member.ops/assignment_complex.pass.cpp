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
    dpl::complex<T> c2(1.5, 2.5);
    c = c2;
    assert(c.real() == 1.5);
    assert(c.imag() == 2.5);
    dpl::complex<X> c3(3.5, -4.5);
    c = c3;
    assert(c.real() == 3.5);
    assert(c.imag() == -4.5);
}

ONEDPL_TEST_NUM_MAIN
{
    test<float, float>();
    RUN_IF_DOUBLE_SUPPORT(test<float, double>())
    RUN_IF_LDOUBLE_SUPPORT(test<float, long double>())

    RUN_IF_DOUBLE_SUPPORT(test<double, float>())
    RUN_IF_DOUBLE_SUPPORT(test<double, double>())
    RUN_IF_LDOUBLE_SUPPORT(test<double, long double>())

    RUN_IF_LDOUBLE_SUPPORT(test<long double, float>())
    RUN_IF_LDOUBLE_SUPPORT(test<long double, double>())
    RUN_IF_LDOUBLE_SUPPORT(test<long double, long double>())
}
