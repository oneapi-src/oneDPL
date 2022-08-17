//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <complex>

// template<Arithmetic T>
//   T
//   arg(T x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(T x, typename std::enable_if<std::is_integral<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(dpl::arg(x)), double>::value), "");
    assert(dpl::arg(x) == dpl::arg(dpl::complex<double>(static_cast<double>(x), 0)));
}

template <class T>
void
test(T x, typename std::enable_if<!std::is_integral<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(dpl::arg(x)), T>::value), "");
    assert(dpl::arg(x) == dpl::arg(dpl::complex<T>(x, 0)));
}

template <class T>
void
test()
{
    test<T>(0);
    test<T>(1);
    test<T>(10);
}

template <typename EnableDouble, typename EnableLongDouble>
void
run_test()
{
    test<float>();
    oneapi::dpl::__internal::__invoke_if(EnableDouble{}, [&]() { test<double>(); });
    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<long double>(); });
    test<int>();
    test<unsigned>();
    test<long long>();
}

int main(int, char**)
{
    // Run on host
    run_test<::std::true_type, ::std::true_type>();

    // Run test in Kernel
    TestUtils::run_test_in_kernel([&]() { run_test<::std::true_type, ::std::false_type>(); },
                                  [&]() { run_test<::std::false_type, ::std::false_type>(); });

    return TestUtils::done();
}
