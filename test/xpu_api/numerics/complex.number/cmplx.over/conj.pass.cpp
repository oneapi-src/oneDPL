//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>      complex<T>           conj(const complex<T>&);
//                        complex<long double> conj(long double);
//                        complex<double>      conj(double);
// template<Integral T>   complex<double>      conj(T);
//                        complex<float>       conj(float);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(T x, ::std::enable_if_t<std::is_integral_v<T>>* = 0)
{
    static_assert((std::is_same_v<decltype(dpl::conj(x)), dpl::complex<double> >));
    assert(dpl::conj(x) == dpl::conj(dpl::complex<double>(x, 0)));
}

template <class T>
void
test(T x, ::std::enable_if_t<std::is_floating_point_v<T>>* = 0)
{
    static_assert((std::is_same_v<decltype(dpl::conj(x)), dpl::complex<T>>));
    assert(dpl::conj(x) == dpl::conj(dpl::complex<T>(x, 0)));
}

template <class T>
void
test(T x, ::std::enable_if_t<!std::is_integral_v<T> && !std::is_floating_point_v<T>>* = 0)
{
    static_assert((std::is_same_v<decltype(dpl::conj(x)), dpl::complex<T>>));
    assert(dpl::conj(x) == dpl::conj(dpl::complex<T>(x, 0)));
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
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    // This check required to avoid code with dpl::complex<double> instantiation
    // when double type not supported on device
    IF_DOUBLE_SUPPORT(test<int>())
    IF_DOUBLE_SUPPORT(test<unsigned>())
    IF_DOUBLE_SUPPORT(test<long long>())

  return 0;
}
