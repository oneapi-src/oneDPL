//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   bool
//   operator==(const T& lhs, const complex<T>& rhs);

#include "support/test_complex.h"

template <class T>
void
test_constexpr()
{
    {
    constexpr T lhs(-2.5f);
    constexpr dpl::complex<T> rhs(1.5f,  2.5f);
    STD_COMPLEX_TESTS_STATIC_ASSERT(!(lhs == rhs));
    }
    {
    constexpr T lhs(-2.5f);
    constexpr dpl::complex<T> rhs(1.5f,  0);
    STD_COMPLEX_TESTS_STATIC_ASSERT(!(lhs == rhs));
    }
    {
    constexpr T lhs(1.5f);
    constexpr dpl::complex<T> rhs(1.5f, 2.5f);
    STD_COMPLEX_TESTS_STATIC_ASSERT(!(lhs == rhs));
    }
    {
    constexpr T lhs(1.5f);
    constexpr dpl::complex<T> rhs(1.5f, 0);
    STD_COMPLEX_TESTS_STATIC_ASSERT(lhs == rhs);
    }
}

template <class T>
void
test()
{
    {
    T lhs(-2.5f);
    dpl::complex<T> rhs(1.5f,  2.5f);
    assert(!(lhs == rhs));
    }
    {
    T lhs(-2.5f);
    dpl::complex<T> rhs(1.5f,  0);
    assert(!(lhs == rhs));
    }
    {
    T lhs(1.5f);
    dpl::complex<T> rhs(1.5f, 2.5f);
    assert(!(lhs == rhs));
    }
    {
    T lhs(1.5f);
    dpl::complex<T> rhs(1.5f, 0);
    assert(lhs == rhs);
    }

    test_constexpr<T> ();
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    //     test_constexpr<int>();

  return 0;
}
