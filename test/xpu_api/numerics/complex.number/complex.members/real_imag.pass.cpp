//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// void real(T val);
// void imag(T val);

#include "support/test_complex.h"

template <class T>
void
test_constexpr()
{
#if TEST_STD_VER > 11
    constexpr dpl::complex<T> c1;
    static_assert(c1.real() == 0, "");
    static_assert(c1.imag() == 0, "");
    constexpr dpl::complex<T> c2(3);
    static_assert(c2.real() == 3, "");
    static_assert(c2.imag() == 0, "");
    constexpr dpl::complex<T> c3(3, 4);
    static_assert(c3.real() == 3, "");
    static_assert(c3.imag() == 4, "");
#endif
}

template <class T>
void
test()
{
    dpl::complex<T> c;
    assert(c.real() == 0);
    assert(c.imag() == 0);
    c.real(3.5);
    assert(c.real() == 3.5);
    assert(c.imag() == 0);
    c.imag(4.5);
    assert(c.real() == 3.5);
    assert(c.imag() == 4.5);
    c.real(-4.5);
    assert(c.real() == -4.5);
    assert(c.imag() == 4.5);
    c.imag(-5.5);
    assert(c.real() == -4.5);
    assert(c.imag() == -5.5);

    test_constexpr<T> ();
}

template <typename EnableDouble, typename EnableLongDouble>
void
run_test()
{
    test<float>();
    oneapi::dpl::__internal::__invoke_if(EnableDouble{}, [&]() { test<double>(); });
    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<long double>(); });
    test_constexpr<int>();
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
