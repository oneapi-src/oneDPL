//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <complex>

#include "support/test_complex.h"

ONEDPL_TEST_NUM_MAIN
{
    using namespace std;

    IF_LONG_DOUBLE_SUPPORT(dpl::complex<long double> c1 = 3.0il;
                           assert(c1 == dpl::complex<long double>(0, 3.0));
                           auto c2 = 3il;
                           assert(c1 == c2))

    IF_LONG_DOUBLE_SUPPORT(dpl::complex<double> c1 = 3.0i;
                           assert(c1 == dpl::complex<double>(0, 3.0));
                           auto c2 = 3i;
                           assert(c1 == c2))

    IF_LONG_DOUBLE_SUPPORT(dpl::complex<float> c1 = 3.0if;
                           assert ( c1 == dpl::complex<float>(0, 3.0));
                           auto c2 = 3if;
                           assert ( c1 == c2 ))

  return 0;
}
