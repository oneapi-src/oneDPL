//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<> class complex<double>
// {
// public:
//     constexpr complex(const complex<float>&);
// };

#include "support/test_complex.h"

ONEDPL_TEST_NUM_MAIN
{
    RUN_IF_DOUBLE_SUPPORT(const dpl::complex<float> cd(2.5, 3.5);
                          dpl::complex<double> cf = cd;
                          assert(cf.real() == cd.real());
                          assert(cf.imag() == cd.imag()))

#if TEST_STD_VER >= 11
    RUN_IF_DOUBLE_SUPPORT(constexpr dpl::complex<float> cd(2.5, 3.5);
                          constexpr dpl::complex<double> cf = cd;
                          static_assert(cf.real() == cd.real(), "");
                          static_assert(cf.imag() == cd.imag(), ""))
#endif
}
