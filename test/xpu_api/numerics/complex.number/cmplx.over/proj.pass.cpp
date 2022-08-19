//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>    complex<T>           proj(const complex<T>&);
//                      complex<long double> proj(long double);
//                      complex<double>      proj(double);
// template<Integral T> complex<double>      proj(T);
//                      complex<float>       proj(float);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(T x, typename std::enable_if<std::is_integral<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(dpl::proj(x)), dpl::complex<double> >::value), "");
    assert(dpl::proj(x) == dpl::proj(dpl::complex<double>(x, 0)));
}

template <class T>
void
test(T x, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(dpl::proj(x)), dpl::complex<T> >::value), "");
    assert(dpl::proj(x) == dpl::proj(dpl::complex<T>(x, 0)));
}

template <class T>
void
test(T x, typename std::enable_if<!std::is_integral<T>::value &&
                                  !std::is_floating_point<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(dpl::proj(x)), dpl::complex<T> >::value), "");
    assert(dpl::proj(x) == dpl::proj(dpl::complex<T>(x, 0)));
}

template <class T>
void
test()
{
    test<T>(0);
    test<T>(1);
    test<T>(10);
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    RUN_IF_DOUBLE_SUPPORT(test<double>())
    RUN_IF_LDOUBLE_SUPPORT(test<long double>())
    test<int>();
    test<unsigned>();
    test<long long>();

  return 0;
}
