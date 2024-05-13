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

template <class TChecker>
void test_edges(TChecker& check_obj)
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::proj(testcases[i]);
        switch (classify(testcases[i]))
        {
        case zero:
        case non_zero:
            CALL_CHECK_OBJ_I(check_obj, i, r == testcases[i]);
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(dpl::real(r)) == std::signbit(dpl::real(testcases[i])));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(dpl::imag(r)) == std::signbit(dpl::imag(testcases[i])));
            break;
        case inf:
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(dpl::real(r)) && dpl::real(r) > 0);
            CALL_CHECK_OBJ_I(check_obj, i, dpl::imag(r) == 0);
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(dpl::imag(r)) == std::signbit(dpl::imag(testcases[i])));
            break;
        case NaN:
        case non_zero_nan:
            CALL_CHECK_OBJ_I(check_obj, i, classify(r) == classify(testcases[i]));
            break;
        }
    }
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    IF_DOUBLE_SUPPORT_REF_CAPT(test_edges(check_obj))

  return 0;
}
