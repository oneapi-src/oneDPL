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
//   cosh(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& c, dpl::complex<T> x)
{
    assert(dpl::cosh(c) == x);
}

template <class T>
void
test()
{
    test(dpl::complex<T>(0, 0), dpl::complex<T>(1, 0));
}

template <class TChecker>
void test_edges(TChecker& check_obj)
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::cosh(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            CALL_CHECK_OBJ_I(check_obj, i, r.real() == 1);
            CALL_CHECK_OBJ_I(check_obj, i, r.imag() == 0);
#ifndef _PSTL_ICC_TEST_COMPLEX_COSH_MINUS_ZERO_MINUS_ZERO_BROKEN_SIGNBIT
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
#endif // _PSTL_ICC_TEST_COMPLEX_COSH_MINUS_ZERO_MINUS_ZERO_BROKEN_SIGNBIT
        }
#if !_PSTL_TEST_COMPLEX_COSH_BROKEN
        else if (testcases[i].real() == 0 && std::isinf(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, r.imag() == 0);
        }
        else if (testcases[i].real() == 0 && std::isnan(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, r.imag() == 0);
        }
#endif // _PSTL_TEST_COMPLEX_COSH_BROKEN
        else if (std::isfinite(testcases[i].real()) && std::isinf(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
        }
        else if (std::isfinite(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
        }
#if !_PSTL_TEST_COMPLEX_COSH_BROKEN
        else if (std::isinf(testcases[i].real()) && testcases[i].imag() == 0)
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, !std::signbit(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, r.imag() == 0);
#ifndef _PSTL_ICC_TEST_COMPLEX_COSH_MINUS_INF_MINUS_ZERO_BROKEN_SIGNBIT
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
#endif // _PSTL_ICC_TEST_COMPLEX_COSH_MINUS_INF_MINUS_ZERO_BROKEN_SIGNBIT
        }
        else if (std::isinf(testcases[i].real()) && std::isfinite(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(r.real()) == std::signbit(std::cos(testcases[i].imag())));
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.imag()));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(r.imag()) == std::signbit(testcases[i].real() * dpl::sin(testcases[i].imag())));
        }
        else if (std::isinf(testcases[i].real()) && std::isinf(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, r.real() > 0);
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
        }
        else if (std::isnan(testcases[i].real()) && testcases[i].imag() == 0)
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, r.imag() == 0);
        }
#endif // _PSTL_TEST_COMPLEX_COSH_BROKEN
        else if (std::isnan(testcases[i].real()) && std::isfinite(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
        }
        else if (std::isnan(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
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
