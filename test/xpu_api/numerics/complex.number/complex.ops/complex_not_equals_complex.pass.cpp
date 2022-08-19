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
//   operator!=(const complex<T>& lhs, const complex<T>& rhs);

#include "support/test_complex.h"

template <class T>
void
test_constexpr()
{
#if TEST_STD_VER > 11
    {
    constexpr dpl::complex<T> lhs(1.5,  2.5);
    constexpr dpl::complex<T> rhs(1.5, -2.5);
    static_assert(lhs != rhs, "");
    }
    {
    constexpr dpl::complex<T> lhs(1.5, 2.5);
    constexpr dpl::complex<T> rhs(1.5, 2.5);
    static_assert(!(lhs != rhs), "" );
    }
#endif
}


template <class T>
void
test()
{
    {
    dpl::complex<T> lhs(1.5,  2.5);
    dpl::complex<T> rhs(1.5, -2.5);
    assert(lhs != rhs);
    }
    {
    dpl::complex<T> lhs(1.5, 2.5);
    dpl::complex<T> rhs(1.5, 2.5);
    assert(!(lhs != rhs));
    }

    test_constexpr<T> ();
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    oneapi::dpl::__internal::__invoke_if(EnableDouble{}, [&]() { test<double>(); });
    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<long double>(); });
    //   test_constexpr<int> ();
}
