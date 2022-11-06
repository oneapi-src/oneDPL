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
//   operator!=(const complex<T>& lhs, const T& rhs);

#include "support/test_complex.h"

template <class T>
void
test_constexpr()
{
    {
    constexpr dpl::complex<T> lhs(1.5,  2.5);
    constexpr T rhs(-2.5);
    STD_COMPLEX_TESTS_STATIC_ASSERT(lhs != rhs, "");
    }
    {
    constexpr dpl::complex<T> lhs(1.5,  0);
    constexpr T rhs(-2.5);
    STD_COMPLEX_TESTS_STATIC_ASSERT(lhs != rhs, "");
    }
    {
    constexpr dpl::complex<T> lhs(1.5, 2.5);
    constexpr T rhs(1.5);
    STD_COMPLEX_TESTS_STATIC_ASSERT(lhs != rhs, "");
    }
    {
    constexpr dpl::complex<T> lhs(1.5, 0);
    constexpr T rhs(1.5);
    STD_COMPLEX_TESTS_STATIC_ASSERT( !(lhs != rhs), "");
    }
}

template <class T>
void
test()
{
    {
    dpl::complex<T> lhs(1.5,  2.5);
    T rhs(-2.5);
    assert(lhs != rhs);
    }
    {
    dpl::complex<T> lhs(1.5,  0);
    T rhs(-2.5);
    assert(lhs != rhs);
    }
    {
    dpl::complex<T> lhs(1.5, 2.5);
    T rhs(1.5);
    assert(lhs != rhs);
    }
    {
    dpl::complex<T> lhs(1.5, 0);
    T rhs(1.5);
    assert( !(lhs != rhs));
    }

    test_constexpr<T> ();
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    //     test_constexpr<int> ();

  return 0;
}
