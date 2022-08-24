//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// constexpr complex(const T& re = T(), const T& im = T());

#include "support/test_complex.h"

template <class T>
void
test()
{
    {
    const dpl::complex<T> c;
    assert(c.real() == 0);
    assert(c.imag() == 0);
    }
    {
    const dpl::complex<T> c = 7.5;
    assert(c.real() == 7.5);
    assert(c.imag() == 0);
    }
    {
    const dpl::complex<T> c(8.5);
    assert(c.real() == 8.5);
    assert(c.imag() == 0);
    }
    {
    const dpl::complex<T> c(10.5, -9.5);
    assert(c.real() == 10.5);
    assert(c.imag() == -9.5);
    }
#if TEST_STD_VER >= 11
    {
    constexpr dpl::complex<T> c;
    static_assert(c.real() == 0, "");
    static_assert(c.imag() == 0, "");
    }
    {
    constexpr dpl::complex<T> c = 7.5;
    static_assert(c.real() == 7.5, "");
    static_assert(c.imag() == 0, "");
    }
    {
    constexpr dpl::complex<T> c(8.5);
    static_assert(c.real() == 8.5, "");
    static_assert(c.imag() == 0, "");
    }
    {
    constexpr dpl::complex<T> c(10.5, -9.5);
    static_assert(c.real() == 10.5, "");
    static_assert(c.imag() == -9.5, "");
    }
#endif
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT_IN_RUNTIME(test<double>())
    IF_CAN_COMPILE_LONG_DOUBLE(test<long double>())

  return 0;
}
