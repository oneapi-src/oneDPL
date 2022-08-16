//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

#include "support/test_complex.h"

template <class T>
void
test()
{
    dpl::complex<T> z;
    T* a = (T*)&z;
    assert(0 == z.real());
    assert(0 == z.imag());
    assert(a[0] == z.real());
    assert(a[1] == z.imag());
    a[0] = 5;
    a[1] = 6;
    assert(a[0] == z.real());
    assert(a[1] == z.imag());
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
