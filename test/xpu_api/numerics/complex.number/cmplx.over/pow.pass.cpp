//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <complex>

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const T& x, const complex<U>& y);

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const complex<T>& x, const U& y);

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const complex<T>& x, const complex<U>& y);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
double
promote(T, typename std::enable_if<std::is_integral<T>::value>::type* = 0);

float promote(float);
double promote(double);
long double promote(long double);

template <class T, class U>
void
test(T x, const dpl::complex<U>& y)
{
    typedef decltype(promote(x) + promote(dpl::real(y))) V;
    static_assert((std::is_same<decltype(dpl::pow(x, y)), dpl::complex<V> >::value), "");
    assert(dpl::pow(x, y) == dpl::pow(dpl::complex<V>(x, 0), dpl::complex<V>(y)));
}

template <class T, class U>
void
test(const dpl::complex<T>& x, U y)
{
    typedef decltype(promote(dpl::real(x)) + promote(y)) V;
    static_assert((std::is_same<decltype(dpl::pow(x, y)), dpl::complex<V> >::value), "");
    assert(dpl::pow(x, y) == dpl::pow(dpl::complex<V>(x), dpl::complex<V>(y, 0)));
}

template <class T, class U>
void
test(const dpl::complex<T>& x, const dpl::complex<U>& y)
{
    typedef decltype(promote(dpl::real(x)) + promote(dpl::real(y))) V;
    static_assert((std::is_same<decltype(dpl::pow(x, y)), dpl::complex<V> >::value), "");
    assert(dpl::pow(x, y) == dpl::pow(dpl::complex<V>(x), dpl::complex<V>(y)));
}

template <class T, class U>
void
test(typename std::enable_if<std::is_integral<T>::value>::type* = 0, typename std::enable_if<!std::is_integral<U>::value>::type* = 0)
{
    test(T(3), dpl::complex<U>(4, 5));
    test(dpl::complex<U>(3, 4), T(5));
}

template <class T, class U>
void
test(typename std::enable_if<!std::is_integral<T>::value>::type* = 0, typename std::enable_if<!std::is_integral<U>::value>::type* = 0)
{
    test(T(3), dpl::complex<U>(4, 5));
    test(dpl::complex<T>(3, 4), U(5));
    test(dpl::complex<T>(3, 4), dpl::complex<U>(5, 6));
}

ONEDPL_TEST_NUM_MAIN
{
    test<int, float>();
    INVOKE_IF_DOUBLE_SUPPORT(test<int, double>())
    INVOKE_IF_LONG_DOUBLE_SUPPORT(test<int, long double>())

    test<unsigned, float>();
    INVOKE_IF_DOUBLE_SUPPORT(test<unsigned, double>())
    INVOKE_IF_LONG_DOUBLE_SUPPORT(test<unsigned, long double>())

    test<long long, float>();
    INVOKE_IF_DOUBLE_SUPPORT(test<long long, double>())
    INVOKE_IF_LONG_DOUBLE_SUPPORT(test<long long, long double>())

    INVOKE_IF_DOUBLE_SUPPORT(test<float, double>())
    INVOKE_IF_LONG_DOUBLE_SUPPORT(test<float, long double>())

    INVOKE_IF_DOUBLE_SUPPORT(test<double, float>())
    INVOKE_IF_LONG_DOUBLE_SUPPORT(test<double, long double>())

    INVOKE_IF_LONG_DOUBLE_SUPPORT(test<long double, float>())
    INVOKE_IF_LONG_DOUBLE_SUPPORT(test<long double, double>())

  return 0;
}
