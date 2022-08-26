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
//   conj(const complex<T>& x);

#include "support/test_complex.h"

template <class T>
void
test(const dpl::complex<T>& z, dpl::complex<T> x)
{
    assert(dpl::conj(z) == x);
}

template <class T>
void
test()
{
    test(dpl::complex<T>(1, 2), dpl::complex<T>(1, -2));
    test(dpl::complex<T>(-1, 2), dpl::complex<T>(-1, -2));
    test(dpl::complex<T>(1, -2), dpl::complex<T>(1, 2));
    test(dpl::complex<T>(-1, -2), dpl::complex<T>(-1, 2));
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())

  return 0;
}
