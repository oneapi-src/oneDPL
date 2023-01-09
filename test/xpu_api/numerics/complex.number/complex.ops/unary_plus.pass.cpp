//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   complex<T>
//   operator+(const complex<T>&);

#include "support/test_complex.h"

template <class T>
void
test()
{
    dpl::complex<T> z(1.5f, 2.5f);
    assert(z.real() == 1.5f);
    assert(z.imag() == 2.5f);
    dpl::complex<T> c = +z;
    assert(c.real() == 1.5f);
    assert(c.imag() == 2.5f);
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())

  return 0;
}
