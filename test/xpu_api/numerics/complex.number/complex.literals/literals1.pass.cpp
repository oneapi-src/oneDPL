//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <chrono>

#include "support/test_complex.h"

template <typename EnableDouble, typename EnableLongDouble>
void
run_test()
{
    using namespace std::literals;

    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{},
                                         [&]()
                                         {
                                             dpl::complex<long double> c1 = 3.0il;
                                             assert(c1 == dpl::complex<long double>(0, 3.0));
                                             auto c2 = 3il;
                                             assert(c1 == c2);
                                         });

    oneapi::dpl::__internal::__invoke_if(EnableDouble{},
                                         [&]()
                                         {
                                             dpl::complex<double> c1 = 3.0i;
                                             assert(c1 == dpl::complex<double>(0, 3.0));
                                             auto c2 = 3i;
                                             assert(c1 == c2);
                                         });

    {
    dpl::complex<float> c1 = 3.0if;
    assert ( c1 == dpl::complex<float>(0, 3.0));
    auto c2 = 3if;
    assert ( c1 == c2 );
    }
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
