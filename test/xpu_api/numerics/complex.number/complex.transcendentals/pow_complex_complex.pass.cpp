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
//   pow(const complex<T>& x, const complex<T>& y);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& a, const dpl::complex<T>& b, dpl::complex<T> x)
{
    dpl::complex<T> c = dpl::pow(a, b);
    is_about(dpl::real(c), dpl::real(x));
    is_about(dpl::imag(c), dpl::imag(x));
}

template <class T>
void
test()
{
    test(dpl::complex<T>(2, 3), dpl::complex<T>(2, 0), dpl::complex<T>(-5, 12));
}

void test_edges()
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        for (unsigned j = 0; j < N; ++j)
        {
            dpl::complex<double> r = dpl::pow(testcases[i], testcases[j]);
            dpl::complex<double> z = dpl::exp(testcases[j] * dpl::log(testcases[i]));
            if (std::isnan(dpl::real(r)))
                assert(std::isnan(dpl::real(z)));
            else
            {
                assert(dpl::real(r) == dpl::real(z));
                assert(std::signbit(dpl::real(r)) == std::signbit(dpl::real(z)));
            }
            if (std::isnan(dpl::imag(r)))
                assert(std::isnan(dpl::imag(z)));
            else
            {
                assert(dpl::imag(r) == dpl::imag(z));
                assert(std::signbit(dpl::imag(r)) == std::signbit(dpl::imag(z)));
            }
        }
    }
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
#ifndef _PSTL_ICC_TEST_COMPLEX_POW_COMPLEX_COMPLEX_PASS_BROKEN_TEST_EDGES
    IF_DOUBLE_SUPPORT(test_edges())
#endif // _PSTL_ICC_TEST_COMPLEX_POW_COMPLEX_COMPLEX_PASS_BROKEN_TEST_EDGES

  return 0;
}
