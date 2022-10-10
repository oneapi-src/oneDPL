//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <complex>

// template<class T>
//   complex<T>
//   sqrt(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& c, dpl::complex<T> x)
{
// Suppress clang warning: floating-point comparison is always true; constant cannot be represented exactly in type 'float' [-Wliteral-range]
CLANG_DIAGNOSTIC_PUSH
CLANG_DIAGNOSTIC_IGNORED_LITERAL_RANGE

    dpl::complex<T> a = dpl::sqrt(c);
    is_about(dpl::real(a), dpl::real(x));
    assert(dpl::abs(dpl::imag(c)) < 1.e-6);

CLANG_DIAGNOSTIC_POP
}

template <class T>
void
test()
{
    test(dpl::complex<T>(64, 0), dpl::complex<T>(8, 0));
}

void test_edges()
{
// Suppress clang warning: comparison with NaN always evaluates to false in fast floating point modes [-Wtautological-constant-compare]
CLANG_DIAGNOSTIC_PUSH
CLANG_DIAGNOSTIC_IGNORED_AUTOLOGICAL_CONSTANT_COMPARE

    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::sqrt(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            assert(!std::signbit(r.real()));
            assert(std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
        else if (dpl::isinf(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(r.real() > 0);
            assert(dpl::isinf(r.imag()));
            assert(std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
        else if (std::isfinite(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (dpl::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isfinite(testcases[i].imag()))
        {
            assert(r.real() == 0);
            assert(!std::signbit(r.real()));
            assert(dpl::isinf(r.imag()));
            assert(std::signbit(testcases[i].imag()) == std::signbit(r.imag()));
        }
        else if (dpl::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isfinite(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(r.real() > 0);
            assert(r.imag() == 0);
            assert(std::signbit(testcases[i].imag()) == std::signbit(r.imag()));
        }
        else if (dpl::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isnan(testcases[i].imag()))
        {
            assert(std::isnan(r.real()));
            assert(dpl::isinf(r.imag()));
        }
        else if (dpl::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isnan(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(r.real() > 0);
            assert(std::isnan(r.imag()));
        }
        else if (std::isnan(testcases[i].real()) && (std::isfinite(testcases[i].imag()) || std::isnan(testcases[i].imag())))
        {
            assert(std::isnan(r.real()));
            assert(std::isnan(r.imag()));
        }
        else if (std::signbit(testcases[i].imag()))
        {
            assert(!std::signbit(r.real()));
            assert(std::signbit(r.imag()));
        }
        else
        {
            assert(!std::signbit(r.real()));
            assert(!std::signbit(r.imag()));
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
