//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   complex<T>
//   operator-(const T& lhs, const complex<T>& rhs);

#include "support/test_complex.h"

template <class T>
void
test(const T& lhs, const dpl::complex<T>& rhs, dpl::complex<T> x)
{
    assert(lhs - rhs == x);
}

template <class T>
void
test()
{
    {
    T lhs(1.5);
    dpl::complex<T> rhs(3.5, 4.5);
    dpl::complex<T>   x(-2.0, -4.5);
    test(lhs, rhs, x);
    }
    {
    T lhs(1.5);
    dpl::complex<T> rhs(-3.5, 4.5);
    dpl::complex<T>   x(5.0, -4.5);
    test(lhs, rhs, x);
    }
}

template <typename EnableDouble, typename EnableLongDouble>
void
run_test()
{
    test<float>();
    oneapi::dpl::__internal::__invoke_if(EnableDouble{}, [&]() { test<double>(); });
    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<long double>(); });
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
