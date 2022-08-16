//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator*=(const complex& rhs);

#include "support/test_complex.h"

template <class T>
void
test()
{
    dpl::complex<T> c(1);
    const dpl::complex<T> c2(1.5, 2.5);
    assert(c.real() == 1);
    assert(c.imag() == 0);
    c *= c2;
    assert(c.real() == 1.5);
    assert(c.imag() == 2.5);
    c *= c2;
    assert(c.real() == -4);
    assert(c.imag() == 7.5);

    dpl::complex<T> c3;

    c3 = c;
    dpl::complex<int> ic (1,1);
    c3 *= ic;
    assert(c3.real() == -11.5);
    assert(c3.imag() ==   3.5);

    c3 = c;
    dpl::complex<float> fc (1,1);
    c3 *= fc;
    assert(c3.real() == -11.5);
    assert(c3.imag() ==   3.5);
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
