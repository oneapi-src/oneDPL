//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator*=(const T& rhs);

#include "support/test_complex.h"

template <class T>
void
test()
{
    dpl::complex<T> c(1);
    assert(c.real() == 1);
    assert(c.imag() == 0);
    c *= 1.5;
    assert(c.real() == 1.5);
    assert(c.imag() == 0);
    c *= 1.5;
    assert(c.real() == 2.25);
    assert(c.imag() == 0);
    c *= -1.5;
    assert(c.real() == -3.375);
    assert(c.imag() == 0);
    c.imag(2);
    c *= 1.5;
    assert(c.real() == -5.0625);
    assert(c.imag() == 3);
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT_IN_RUNTIME(test<double>())
    IF_LONG_DOUBLE_SUPPORT_IN_COMPILETIME(test<long double>())

  return 0;
}
