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

struct PromoteFloat
{
    static float promote(float);
};

struct PromoteDouble : PromoteFloat
{
    template <class T>
    static double
    promote(T, typename std::enable_if<std::is_integral<T>::value>::type* = 0);

    static double
    promote(double);
};

struct PromoteLongDouble : PromoteDouble
{
    static long double promote(long double);
};

template <class TPromote, class T, class U>
void
test(T x, const dpl::complex<U>& y)
{
    typedef decltype(TPromote::promote(x) + TPromote::promote(dpl::real(y))) V;
    static_assert((std::is_same<decltype(dpl::pow(x, y)), dpl::complex<V> >::value), "");
    is_about(dpl::pow(x, y), dpl::pow(dpl::complex<V>(x, 0), dpl::complex<V>(y)));
}

template <class TPromote, class T, class U>
void
test(const dpl::complex<T>& x, U y)
{
    typedef decltype(TPromote::promote(dpl::real(x)) + TPromote::promote(y)) V;
    static_assert((std::is_same<decltype(dpl::pow(x, y)), dpl::complex<V> >::value), "");
    is_about(dpl::pow(x, y), dpl::pow(dpl::complex<V>(x), dpl::complex<V>(y, 0)));
}

template <class TPromote, class T, class U>
void
test(const dpl::complex<T>& x, const dpl::complex<U>& y)
{
    typedef decltype(TPromote::promote(dpl::real(x)) + TPromote::promote(dpl::real(y))) V;
    static_assert((std::is_same<decltype(dpl::pow(x, y)), dpl::complex<V> >::value), "");
    assert(dpl::pow(x, y) == dpl::pow(dpl::complex<V>(x), dpl::complex<V>(y)));
}

template <class TPromote, class T, class U>
void
test(typename std::enable_if<std::is_integral<T>::value>::type* = 0, typename std::enable_if<!std::is_integral<U>::value>::type* = 0)
{
    test<TPromote>(T(3), dpl::complex<U>(4, 5));
    test<TPromote>(dpl::complex<U>(3, 4), T(5));
}

template <class TPromote, class T, class U>
void
test(typename std::enable_if<!std::is_integral<T>::value>::type* = 0, typename std::enable_if<!std::is_integral<U>::value>::type* = 0)
{
    test<TPromote>(T(3), dpl::complex<U>(4, 5));
    test<TPromote>(dpl::complex<T>(3, 4), U(5));
    test<TPromote>(dpl::complex<T>(3, 4), dpl::complex<U>(5, 6));
}

ONEDPL_TEST_NUM_MAIN
{
#if !_PSTL_GLIBCXX_TEST_COMPLEX_POW_BROKEN
    test<int, float, PromoteFloat>();
#endif // !_PSTL_GLIBCXX_TEST_COMPLEX_POW_BROKEN
    test<PromoteFloat, unsigned, float>();
    test<PromoteFloat, long long, float>();

    IF_DOUBLE_SUPPORT(test<PromoteDouble, int, double>();
                      test<PromoteDouble, unsigned, double>();
                      test<PromoteDouble, long long, double>();
                      test<PromoteDouble, float, double>();
                      test<PromoteDouble, double, float>())

    IF_LONG_DOUBLE_SUPPORT(test<PromoteLongDouble, unsigned, long double>();
                           test<PromoteLongDouble, long long, long double>();
                           test<PromoteLongDouble, float, long double>();
                           test<PromoteLongDouble, double, long double>();
                           test<PromoteLongDouble, long double, float>();
                           test<PromoteLongDouble, long double, double>())

  return 0;
}
