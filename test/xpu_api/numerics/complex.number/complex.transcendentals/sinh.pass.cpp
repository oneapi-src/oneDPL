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
//   sinh(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& c, dpl::complex<T> x)
{
    assert(dpl::sinh(c) == x);
}

template <class T>
void
test()
{
    test(dpl::complex<T>(0, 0), dpl::complex<T>(0, 0));
}

void test_edges()
{
// Suppress clang warning: comparison with infinity always evaluates to false in fast floating point modes [-Wtautological-constant-compare]
CLANG_DIAGNOSTIC_PUSH
CLANG_DIAGNOSTIC_IGNORED_AUTOLOGICAL_CONSTANT_COMPARE

    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::sinh(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            assert(r.real() == 0);
            assert(std::signbit(r.real()) == std::signbit(testcases[i].real()));
            assert(r.imag() == 0);
            assert(std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
        else if (testcases[i].real() == 0 && dpl::isinf(testcases[i].imag()))
        {
            assert(r.real() == 0);
            assert(dpl::isnan(r.imag()));
        }
        else if (std::isfinite(testcases[i].real()) && dpl::isinf(testcases[i].imag()))
        {
            assert(dpl::isnan(r.real()));
            assert(dpl::isnan(r.imag()));
        }
        else if (testcases[i].real() == 0 && dpl::isnan(testcases[i].imag()))
        {
            assert(r.real() == 0);
            assert(dpl::isnan(r.imag()));
        }
        else if (std::isfinite(testcases[i].real()) && dpl::isnan(testcases[i].imag()))
        {
            assert(dpl::isnan(r.real()));
            assert(dpl::isnan(r.imag()));
        }
        else if (dpl::isinf(testcases[i].real()) && testcases[i].imag() == 0)
        {
            assert(dpl::isinf(r.real()));
            assert(std::signbit(r.real()) == std::signbit(testcases[i].real()));
            assert(r.imag() == 0);
            assert(std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
        else if (dpl::isinf(testcases[i].real()) && std::isfinite(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(std::signbit(r.real()) == std::signbit(testcases[i].real() * std::cos(testcases[i].imag())));
            assert(dpl::isinf(r.imag()));
            assert(std::signbit(r.imag()) == std::signbit(dpl::sin(testcases[i].imag())));
        }
        else if (dpl::isinf(testcases[i].real()) && dpl::isinf(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(dpl::isnan(r.imag()));
        }
        else if (dpl::isinf(testcases[i].real()) && dpl::isnan(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(dpl::isnan(r.imag()));
        }
        else if (dpl::isnan(testcases[i].real()) && testcases[i].imag() == 0)
        {
            assert(dpl::isnan(r.real()));
            assert(r.imag() == 0);
            assert(std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
        else if (dpl::isnan(testcases[i].real()) && std::isfinite(testcases[i].imag()))
        {
            assert(dpl::isnan(r.real()));
            assert(dpl::isnan(r.imag()));
        }
        else if (dpl::isnan(testcases[i].real()) && dpl::isnan(testcases[i].imag()))
        {
            assert(dpl::isnan(r.real()));
            assert(dpl::isnan(r.imag()));
        }
    }

CLANG_DIAGNOSTIC_POP
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    IF_DOUBLE_SUPPORT(test_edges())

  return 0;
}
