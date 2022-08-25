//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<> class complex<long double>
// {
// public:
//     constexpr complex(const complex<double>&);
// };

#include "support/test_complex.h"

ONEDPL_TEST_NUM_MAIN
{
    IF_LONG_DOUBLE_SUPPORT_IN_COMPILETIME(const dpl::complex<double> cd(2.5, 3.5);
                                          dpl::complex<long double> cf(cd);
                                          assert(cf.real() == cd.real());
                                          assert(cf.imag() == cd.imag()))
#if TEST_STD_VER >= 11
    IF_LONG_DOUBLE_SUPPORT_IN_COMPILETIME(constexpr dpl::complex<double> cd(2.5, 3.5);
                                          constexpr dpl::complex<long double> cf(cd);
                                          static_assert(cf.real() == cd.real(), "");
                                          static_assert(cf.imag() == cd.imag(), ""))
#endif

  return 0;
}
