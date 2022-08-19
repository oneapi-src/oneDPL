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
    dpl::complex<T> c = dpl::pow(a, b);
    is_about(dpl::real(c), dpl::real(x));
    assert(std::abs(dpl::imag(c)) < 1.e-6);
}

template <class T>
void
test()
{
    test(T(2), dpl::complex<T>(2), dpl::complex<T>(4));
}

void test_edges()
{
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
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    oneapi::dpl::__internal::__invoke_if(EnableDouble{}, [&]() { test<double>(); });
    oneapi::dpl::__internal::__invoke_if(EnableLongDouble{}, [&]() { test<long double>(); });
    oneapi::dpl::__internal::__invoke_if(EnableDouble{}, [&]() { test_edges(); });
}
