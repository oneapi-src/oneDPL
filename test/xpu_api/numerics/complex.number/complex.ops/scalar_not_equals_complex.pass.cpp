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
//   operator!=(const T& lhs, const complex<T>& rhs);

#include "support/test_complex.h"

template <class T>
void
test_constexpr()
{
#if TEST_STD_VER > 11
    {
    constexpr T lhs(-2.5);
    constexpr dpl::complex<T> rhs(1.5,  2.5);
    static_assert (lhs != rhs, "");
    }
    {
    constexpr T lhs(-2.5);
    constexpr dpl::complex<T> rhs(1.5,  0);
    static_assert (lhs != rhs, "");
    }
    {
    constexpr T lhs(1.5);
    constexpr dpl::complex<T> rhs(1.5, 2.5);
    static_assert (lhs != rhs, "");
    }
    {
    constexpr T lhs(1.5);
    constexpr dpl::complex<T> rhs(1.5, 0);
    static_assert (!(lhs != rhs), "");
    }
#endif
}

template <class T>
void
test()
{
    {
    T lhs(-2.5);
    dpl::complex<T> rhs(1.5,  2.5);
    assert (lhs != rhs);
    }
    {
    T lhs(-2.5);
    dpl::complex<T> rhs(1.5,  0);
    assert (lhs != rhs);
    }
    {
    T lhs(1.5);
    dpl::complex<T> rhs(1.5, 2.5);
    assert (lhs != rhs);
    }
    {
    T lhs(1.5);
    dpl::complex<T> rhs(1.5, 0);
    assert (!(lhs != rhs));
    }

    test_constexpr<T> ();
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT_IN_RUNTIME(test<double>())
    IF_CAN_COMPILE_LONG_DOUBLE(test<long double>())
    //     test_constexpr<int>();

  return 0;
}
