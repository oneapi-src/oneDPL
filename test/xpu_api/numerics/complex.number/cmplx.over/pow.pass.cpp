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
    typedef decltype(promote(x)+promote(real(y))) V;
    static_assert((std::is_same<decltype(std::pow(x, y)), dpl::complex<V> >::value), "");
    assert(std::pow(x, y) == dpl::pow(dpl::complex<V>(x, 0), dpl::complex<V>(y)));
}

template <class T, class U>
void
test(const dpl::complex<T>& x, U y)
{
    typedef decltype(promote(real(x))+promote(y)) V;
    static_assert((std::is_same<decltype(std::pow(x, y)), dpl::complex<V> >::value), "");
    assert(std::pow(x, y) == dpl::pow(dpl::complex<V>(x), dpl::complex<V>(y, 0)));
}

template <class T, class U>
void
test(const dpl::complex<T>& x, const dpl::complex<U>& y)
{
    typedef decltype(promote(real(x))+promote(real(y))) V;
    static_assert((std::is_same<decltype(std::pow(x, y)), dpl::complex<V> >::value), "");
    assert(std::pow(x, y) == dpl::pow(dpl::complex<V>(x), dpl::complex<V>(y)));
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

template <typename EnableDouble, typename EnableLongDouble>
void
run_test()
{
    test<int, float>();
    oneapi::dpl::__internal::__invoke_if(EnableDouble{}, [&]() { test<int, double>(); });
    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<int, long double>(); });

    test<unsigned, float>();
    oneapi::dpl::__internal::__invoke_if(EnableDouble{}, [&]() { test<unsigned, double>(); });
    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<unsigned, long double>(); });

    test<long long, float>();
    oneapi::dpl::__internal::__invoke_if(EnableDouble{}, [&]() { test<long long, double>(); });
    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<long long, long double>(); });

    oneapi::dpl::__internal::__invoke_if(EnableDouble{}, [&]() { test<float, double>(); });
    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<float, long double>(); });

    oneapi::dpl::__internal::__invoke_if(EnableDouble{}, [&]() { test<double, float>(); });
    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<double, long double>(); });

    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<long double, float>(); });
    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<long double, double>(); });
}

int
main(int, char**)
{
    // Run on host
    run_test<::std::true_type, ::std::true_type>();

    // Run test in Kernel
    TestUtils::run_test_in_kernel([&]() { run_test<::std::true_type, ::std::false_type>(); },
                                  [&]() { run_test<::std::false_type, ::std::false_type>(); });

    return TestUtils::done();
}
