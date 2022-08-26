//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

#include "support/test_complex.h"

template <class T>
void
test()
{
    dpl::complex<T> z;
    T* a = (T*)&z;
    assert(0 == z.real());
    assert(0 == z.imag());
    assert(a[0] == z.real());
    assert(a[1] == z.imag());
    a[0] = 5;
    a[1] = 6;
    assert(a[0] == z.real());
    assert(a[1] == z.imag());
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())

  return 0;
}
