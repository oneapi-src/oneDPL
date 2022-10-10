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
//   acosh(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& c, dpl::complex<T> x)
{
    assert(dpl::acosh(c) == x);
}

template <class T>
void
test()
{
    test(dpl::complex<T>(INFINITY, 1), dpl::complex<T>(INFINITY, 0));
}

void test_edges()
{
// Suppress clang warning: comparison with infinity always evaluates to false in fast floating point modes [-Wtautological-constant-compare]
CLANG_DIAGNOSTIC_PUSH
CLANG_DIAGNOSTIC_IGNORED_AUTOLOGICAL_CONSTANT_COMPARE

    const double pi = std::atan2(+0., -0.);
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::acosh(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            assert(!std::signbit(r.real()));
            if (std::signbit(testcases[i].imag()))
                is_about(r.imag(), -pi/2);
            else
                is_about(r.imag(),  pi/2);
        }
        else if (testcases[i].real() == 1 && testcases[i].imag() == 0)
        {
            assert(r.real() == 0);
            assert(!std::signbit(r.real()));
            assert(r.imag() == 0);
            assert(std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
        else if (testcases[i].real() == -1 && testcases[i].imag() == 0)
        {
            assert(r.real() == 0);
            assert(!std::signbit(r.real()));
            if (std::signbit(testcases[i].imag()))
                is_about(r.imag(), -pi);
            else
                is_about(r.imag(),  pi);
        }
        else if (std::isfinite(testcases[i].real()) && dpl::isinf(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(r.real() > 0);
            if (std::signbit(testcases[i].imag()))
                is_about(r.imag(), -pi/2);
            else
                is_about(r.imag(),  pi/2);
        }
        else if (std::isfinite(testcases[i].real()) && dpl::isnan(testcases[i].imag()))
        {
            assert(dpl::isnan(r.real()));
            assert(dpl::isnan(r.imag()));
        }
        else if (dpl::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isfinite(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(r.real() > 0);
            if (std::signbit(testcases[i].imag()))
                is_about(r.imag(), -pi);
            else
                is_about(r.imag(),  pi);
        }
        else if (dpl::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isfinite(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(r.real() > 0);
            assert(r.imag() == 0);
            assert(std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
        else if (dpl::isinf(testcases[i].real()) && testcases[i].real() < 0 && dpl::isinf(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(r.real() > 0);
            if (std::signbit(testcases[i].imag()))
                is_about(r.imag(), -0.75 * pi);
            else
                is_about(r.imag(),  0.75 * pi);
        }
        else if (dpl::isinf(testcases[i].real()) && testcases[i].real() > 0 && dpl::isinf(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(r.real() > 0);
            if (std::signbit(testcases[i].imag()))
                is_about(r.imag(), -0.25 * pi);
            else
                is_about(r.imag(),  0.25 * pi);
        }
        else if (dpl::isinf(testcases[i].real()) && dpl::isnan(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(r.real() > 0);
            assert(dpl::isnan(r.imag()));
        }
        else if (dpl::isnan(testcases[i].real()) && std::isfinite(testcases[i].imag()))
        {
            assert(dpl::isnan(r.real()));
            assert(dpl::isnan(r.imag()));
        }
        else if (dpl::isnan(testcases[i].real()) && dpl::isinf(testcases[i].imag()))
        {
            assert(dpl::isinf(r.real()));
            assert(r.real() > 0);
            assert(dpl::isnan(r.imag()));
        }
        else if (dpl::isnan(testcases[i].real()) && dpl::isnan(testcases[i].imag()))
        {
            assert(dpl::isnan(r.real()));
            assert(dpl::isnan(r.imag()));
        }
        else
        {
            assert(!std::signbit(r.real()));
            assert(std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
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
