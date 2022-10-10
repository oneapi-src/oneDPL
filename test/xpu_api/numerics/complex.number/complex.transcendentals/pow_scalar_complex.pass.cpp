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
//   pow(const T& x, const complex<T>& y);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const T& a, const dpl::complex<T>& b, dpl::complex<T> x)
{
// Suppress clang warning: floating-point comparison is always true; constant cannot be represented exactly in type 'float' [-Wliteral-range]
CLANG_DIAGNOSTIC_PUSH
CLANG_DIAGNOSTIC_IGNORED_LITERAL_RANGE

    dpl::complex<T> c = dpl::pow(a, b);
    is_about(dpl::real(c), dpl::real(x));
    assert(dpl::abs(dpl::imag(c)) < 1.e-6);

CLANG_DIAGNOSTIC_POP
}

template <class T>
void
test()
{
    test(T(2), dpl::complex<T>(2), dpl::complex<T>(4));
}

void test_edges()
{
// Suppress clang warning: comparison with NaN always evaluates to false in fast floating point modes [-Wtautological-constant-compare]
CLANG_DIAGNOSTIC_PUSH
CLANG_DIAGNOSTIC_IGNORED_AUTOLOGICAL_CONSTANT_COMPARE

    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        for (unsigned j = 0; j < N; ++j)
        {
            dpl::complex<double> r = dpl::pow(dpl::real(testcases[i]), testcases[j]);
            dpl::complex<double> z = dpl::exp(testcases[j] * dpl::log(dpl::complex<double>(dpl::real(testcases[i]))));
            if (std::isnan(dpl::real(r)))
                assert(std::isnan(dpl::real(z)));
            else
            {
                assert(dpl::real(r) == dpl::real(z));
            }
            if (std::isnan(dpl::imag(r)))
                assert(std::isnan(dpl::imag(z)));
            else
            {
                assert(dpl::imag(r) == dpl::imag(z));
            }
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
