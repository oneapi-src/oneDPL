//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   T
//   real(const complex<T>& x);

#include "support/test_complex.h"

template <class T>
void
test()
{
    dpl::complex<T> z(1.5, 2.5);
    assert(dpl::real(z) == 1.5);
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT_IN_RUNTIME(test<double>())
    IF_CAN_COMPILE_LONG_DOUBLE(test<long double>())

  return 0;
}
