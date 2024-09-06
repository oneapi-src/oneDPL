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
//   operator+(const complex<T>& lhs, const T& rhs);

#include "support/test_complex.h"

template <class T>
void
test(const dpl::complex<T>& lhs, const T& rhs, dpl::complex<T> x)
{
    assert(lhs + rhs == x);
}

template <class T>
void
test()
{
    {
    dpl::complex<T> lhs(1.5f, 2.5f);
    T rhs(3.5f);
    dpl::complex<T>   x(5.0f, 2.5f);
    test(lhs, rhs, x);
    }
    {
    dpl::complex<T> lhs(1.5f, -2.5f);
    T rhs(-3.5f);
    dpl::complex<T>   x(-2.0f, -2.5f);
    test(lhs, rhs, x);
    }
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())

  return 0;
}
