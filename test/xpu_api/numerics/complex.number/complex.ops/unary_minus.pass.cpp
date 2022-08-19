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
//   operator-(const complex<T>& lhs);

#include "support/test_complex.h"

template <class T>
void
test()
{
    dpl::complex<T> z(1.5, 2.5);
    assert(z.real() == 1.5);
    assert(z.imag() == 2.5);
    dpl::complex<T> c = -z;
    assert(c.real() == -1.5);
    assert(c.imag() == -2.5);
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    RUN_IF_DOUBLE_SUPPORT(test<double>())
    RUN_IF_LDOUBLE_SUPPORT(test<long double>())
}
