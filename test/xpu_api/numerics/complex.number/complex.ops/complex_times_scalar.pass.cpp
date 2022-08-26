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
//   operator*(const complex<T>& lhs, const T& rhs);

#include "support/test_complex.h"

template <class T>
void
test(const dpl::complex<T>& lhs, const T& rhs, dpl::complex<T> x)
{
    assert(lhs * rhs == x);
}

template <class T>
void
test()
{
    dpl::complex<T> lhs(1.5, 2.5);
    T rhs(1.5);
    dpl::complex<T>   x(2.25, 3.75);
    test(lhs, rhs, x);
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())

  return 0;
}
