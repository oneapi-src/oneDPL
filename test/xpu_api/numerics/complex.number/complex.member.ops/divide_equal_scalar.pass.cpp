//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator/=(const T& rhs);

#include "support/test_complex.h"

template <class T>
void
test()
{
    dpl::complex<T> c(1);
    assert(c.real() == 1);
    assert(c.imag() == 0);
    c /= 0.5;
    assert(c.real() == 2);
    assert(c.imag() == 0);
    c /= 0.5;
    assert(c.real() == 4);
    assert(c.imag() == 0);
    c /= -0.5;
    assert(c.real() == -8);
    assert(c.imag() == 0);
    c.imag(2);
    c /= 0.5;
    assert(c.real() == -16);
    assert(c.imag() == 4);
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())

  return 0;
}
