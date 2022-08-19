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
//   sin(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& c, dpl::complex<T> x)
{
    assert(dpl::sin(c) == x);
}

template <class T>
void
test()
{
    test(dpl::complex<T>(0, 0), dpl::complex<T>(0, 0));
}

void test_edges()
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::sin(testcases[i]);
        dpl::complex<double> t1(-imag(testcases[i]), real(testcases[i]));
        dpl::complex<double> t2 = sinh(t1);
        dpl::complex<double> z(imag(t2), -real(t2));
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

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    RUN_IF_DOUBLE_SUPPORT(test<double>())
    RUN_IF_LDOUBLE_SUPPORT(test<long double>())
    RUN_IF_DOUBLE_SUPPORT(test_edges())

  return 0;
}
