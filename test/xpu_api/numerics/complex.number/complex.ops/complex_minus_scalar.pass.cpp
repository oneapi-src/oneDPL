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
//   operator-(const complex<T>& lhs, const T& rhs);

#include "support/test_complex.h"

template <class T>
void
test(const dpl::complex<T>& lhs, const T& rhs, dpl::complex<T> x)
{
    assert(lhs - rhs == x);
}

template <class T>
void
test()
{
    {
    dpl::complex<T> lhs(1.5, 2.5);
    T rhs(3.5);
    dpl::complex<T>   x(-2.0, 2.5);
    test(lhs, rhs, x);
    }
    {
    dpl::complex<T> lhs(1.5, -2.5);
    T rhs(-3.5);
    dpl::complex<T>   x(5.0, -2.5);
    test(lhs, rhs, x);
    }
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    TestUtils::invoke_test_if(HasDoubleSupportInRuntime(), []() { test<double>(); });
    TestUtils::invoke_test_if(HasLongDoubleSupportInCompiletime(), []() { test<long double>(); });

  return 0;
}
