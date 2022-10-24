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

    TestUtils::invoke_test_if(HasDoubleSupportInRuntime(),
                              []()
                              {
                                  test<float, double>();
                                  test<double, float>();
                                  test<double, double>();
                              });

    TestUtils::invoke_test_if(HasLongDoubleSupportInCompiletime(),
                              []()
                              {
                                  test<float, long double>();
                                  test<double, long double>();
                                  test<long double, float>();
                                  test<long double, double>();
                                  test<long double, long double>();
                              });

  return 0;
}
