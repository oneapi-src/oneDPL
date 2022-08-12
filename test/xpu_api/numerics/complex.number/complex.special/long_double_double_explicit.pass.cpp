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

#include <complex>
#include <cassert>

#include "test_macros.h"

void run_test()
{
    {
    const dpl::complex<double> cd(2.5, 3.5);
    dpl::complex<long double> cf(cd);
    assert(cf.real() == cd.real());
    assert(cf.imag() == cd.imag());
    }
#if TEST_STD_VER >= 11
    {
    constexpr dpl::complex<double> cd(2.5, 3.5);
    constexpr dpl::complex<long double> cf(cd);
    static_assert(cf.real() == cd.real(), "");
    static_assert(cf.imag() == cd.imag(), "");
    }
#endif
}

int main(int, char**)
{
    run_test();

  return 0;
}
