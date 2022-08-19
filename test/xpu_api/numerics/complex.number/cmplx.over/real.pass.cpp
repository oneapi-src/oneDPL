//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<Arithmetic T>
//   T
//   real(const T& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T, int x>
void
test(typename std::enable_if<std::is_integral<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(dpl::real(T(x))), double>::value), "");
    assert(dpl::real(x) == x);
#if TEST_STD_VER > 11
    constexpr T val {x};
    static_assert(dpl::real(val) == val, "");
    constexpr dpl::complex<T> t{val, val};
    static_assert(t.real() == x, "" );
#endif
}

template <class T, int x>
void
test(typename std::enable_if<!std::is_integral<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(dpl::real(T(x))), T>::value), "");
    assert(dpl::real(x) == x);
#if TEST_STD_VER > 11
    constexpr T val {x};
    static_assert(dpl::real(val) == val, "");
    constexpr dpl::complex<T> t{val, val};
    static_assert(t.real() == x, "" );
#endif
}

template <class T>
void
test()
{
    test<T, 0>();
    test<T, 1>();
    test<T, 10>();
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    oneapi::dpl::__internal::__invoke_if(EnableDouble{}, [&]() { test<double>(); });
    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<long double>(); });
    test<int>();
    test<unsigned>();
    test<long long>();
}
