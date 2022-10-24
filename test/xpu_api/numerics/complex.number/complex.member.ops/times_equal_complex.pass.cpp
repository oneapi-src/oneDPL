//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator*=(const complex& rhs);

#include "support/test_complex.h"

template <class T>
void
test()
{
    dpl::complex<T> c(1);
    const dpl::complex<T> c2(1.5, 2.5);
    assert(c.real() == 1);
    assert(c.imag() == 0);
    c *= c2;
    assert(c.real() == 1.5);
    assert(c.imag() == 2.5);
    c *= c2;
    assert(c.real() == -4);
    assert(c.imag() == 7.5);

    dpl::complex<T> c3;

    // Commented to avoid compile error: no viable conversion from 'std::complex<int>' to 'const std::complex<float>::_ComplexT' (aka 'const _Complex float')
    //c3 = c;
    //dpl::complex<int> ic (1,1);
    //c3 *= ic;
    //assert(c3.real() == -11.5);
    //assert(c3.imag() ==   3.5);

    c3 = c;
    dpl::complex<float> fc (1,1);
    c3 *= fc;
    assert(c3.real() == -11.5);
    assert(c3.imag() ==   3.5);
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())

  return 0;
}
