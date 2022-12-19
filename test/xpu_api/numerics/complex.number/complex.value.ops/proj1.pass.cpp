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
//   proj(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& z, dpl::complex<T> x)
{
    assert(proj(z) == x);
}

template <class T>
void
test()
{
    test(dpl::complex<T>(1, 2), dpl::complex<T>(1, 2));
    test(dpl::complex<T>(-1, 2), dpl::complex<T>(-1, 2));
    test(dpl::complex<T>(1, -2), dpl::complex<T>(1, -2));
    test(dpl::complex<T>(-1, -2), dpl::complex<T>(-1, -2));
}

void test_edges()
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::proj(testcases[i]);
        switch (classify(testcases[i]))
        {
        case zero:
        case non_zero:
            assert(r == testcases[i]);
            assert(std::signbit(dpl::real(r)) == std::signbit(dpl::real(testcases[i])));
            assert(std::signbit(dpl::imag(r)) == std::signbit(dpl::imag(testcases[i])));
            break;
        case inf:
            assert(std::isinf(dpl::real(r)) && dpl::real(r) > 0);
            assert(dpl::imag(r) == 0);
            assert(std::signbit(dpl::imag(r)) == std::signbit(dpl::imag(testcases[i])));
            break;
        case NaN:
        case non_zero_nan:
            assert(classify(r) == classify(testcases[i]));
            break;
        }
    }
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    IF_DOUBLE_SUPPORT(test_edges())

  return 0;
}
