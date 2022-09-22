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
//   operator==(const complex<T>& lhs, const T& rhs);

#include "support/test_complex.h"

template <class T>
void
test_constexpr()
{
    {
    constexpr dpl::complex<T> lhs(1.5, 2.5);
    constexpr T rhs(-2.5);
    static_assert(!(lhs == rhs), "");
    }
    {
    constexpr dpl::complex<T> lhs(1.5, 0);
    constexpr T rhs(-2.5);
    static_assert(!(lhs == rhs), "");
    }
    {
    constexpr dpl::complex<T> lhs(1.5, 2.5);
    constexpr T rhs(1.5);
    static_assert(!(lhs == rhs), "");
    }
    {
    constexpr dpl::complex<T> lhs(1.5, 0);
    constexpr T rhs(1.5);
    static_assert( (lhs == rhs), "");
    }
}

template <class T>
void
test()
{
    {
    dpl::complex<T> lhs(1.5,  2.5);
    T rhs(-2.5);
    assert(!(lhs == rhs));
    }
    {
    dpl::complex<T> lhs(1.5, 0);
    T rhs(-2.5);
    assert(!(lhs == rhs));
    }
    {
    dpl::complex<T> lhs(1.5, 2.5);
    T rhs(1.5);
    assert(!(lhs == rhs));
    }
    {
    dpl::complex<T> lhs(1.5, 0);
    T rhs(1.5);
    assert( (lhs == rhs));
    }

    test_constexpr<T> ();
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    TestUtils::invoke_test_if(HasDoubleSupportInRuntime(), []() { test<double>(); });
    TestUtils::invoke_test_if(HasLongDoubleSupportInCompiletime(), []() { test<long double>(); });
    //     test_constexpr<int> ();

  return 0;
}
