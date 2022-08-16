//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<> class complex<float>
// {
// public:
//     explicit constexpr complex(const complex<double>&);
// };

#include "support/test_complex.h"

template <typename EnableDouble, typename EnableLongDouble>
void
run_test()
{
    oneapi::dpl::__internal::__invoke_if(EnableDouble{},
                                         [&]()
                                         {
                                             const dpl::complex<double> cd(2.5, 3.5);
                                             dpl::complex<float> cf(cd);
                                             assert(cf.real() == cd.real());
                                             assert(cf.imag() == cd.imag());
                                         });
#if TEST_STD_VER >= 11
    oneapi::dpl::__internal::__invoke_if(EnableDouble{},
                                         [&]()
                                         {
                                             constexpr dpl::complex<double> cd(2.5, 3.5);
                                             constexpr dpl::complex<float> cf(cd);
                                             static_assert(cf.real() == cd.real(), "");
                                             static_assert(cf.imag() == cd.imag(), "");
                                         });
#endif
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
