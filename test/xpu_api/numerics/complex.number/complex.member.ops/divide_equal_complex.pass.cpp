//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator/=(const complex& rhs);

#include "support/test_complex.h"
#include "./../cases.h"

template <class T>
void
test()
{
    dpl::complex<T> c(-4, 7.5);
    const dpl::complex<T> c2(1.5, 2.5);
    assert(c.real() == -4);
    assert(c.imag() == 7.5);
    c /= c2;
    is_about(c.real(), (T)1.5);
    is_about(c.imag(), (T)2.5);
    c /= c2;
    is_about(c.real(), (T)1.0);
    is_about(c.imag(), (T)0.0);

    dpl::complex<T> c3;

#if !_PSTL_GLIBCXX_TEST_COMPLEX_DIV_EQ_BROKEN
    c3 = c;
    dpl::complex<int> ic (1,1);
    c3 /= ic;
    is_about(c3.real(), (T) 0.5);
    is_about(c3.imag(), (T)-0.5);
#endif // !_PSTL_GLIBCXX_TEST_COMPLEX_DIV_EQ_BROKEN

    c3 = c;
    dpl::complex<float> fc (1,1);
    c3 /= fc;
    is_about(c3.real(), (T) 0.5);
    is_about(c3.imag(), (T)-0.5);
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())

  return 0;
}
