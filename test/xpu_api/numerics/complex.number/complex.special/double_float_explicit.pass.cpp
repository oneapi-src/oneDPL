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
    IF_DOUBLE_SUPPORT(const dpl::complex<float> cd(2.5f, 3.5f);
                      dpl::complex<double> cf(cd);
                      assert(cf.real() == cd.real());
                      assert(cf.imag() == cd.imag()))

    IF_DOUBLE_SUPPORT(constexpr dpl::complex<float> cd(2.5f, 3.5f);
                      constexpr dpl::complex<double> cf(cd);
                      STD_COMPLEX_TESTS_STATIC_ASSERT(cf.real() == cd.real());
                      STD_COMPLEX_TESTS_STATIC_ASSERT(cf.imag() == cd.imag()))

  return 0;
}
