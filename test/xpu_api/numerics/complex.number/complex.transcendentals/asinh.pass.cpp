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
//   asinh(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& c, dpl::complex<T> x)
{
    assert(dpl::asinh(c) == x);
}

template <class T>
void
test()
{
    test(dpl::complex<T>(0, 0), dpl::complex<T>(0, 0));
}

template <class TChecker>
void test_edges(TChecker& check_obj)
{
    const double pi = std::atan2(+0., -0.);
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::asinh(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(r.real()) == std::signbit(testcases[i].real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
        else if (testcases[i].real() == 0 && std::abs(testcases[i].imag()) == 1)
        {
            CALL_CHECK_OBJ_I(check_obj, i, r.real() == 0);
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(testcases[i].imag()) == std::signbit(r.imag()));
            if (std::signbit(testcases[i].imag()))
                CALL_CHECK_OBJ_I(check_obj, i, is_about(r.imag(), -pi / 2));
            else
                CALL_CHECK_OBJ_I(check_obj, i, is_about(r.imag(), pi / 2));
        }
        else if (std::isfinite(testcases[i].real()) && std::isinf(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(testcases[i].real()) == std::signbit(r.real()));
            if (std::signbit(testcases[i].imag()))
                CALL_CHECK_OBJ_I(check_obj, i, is_about(r.imag(), -pi / 2));
            else
                CALL_CHECK_OBJ_I(check_obj, i, is_about(r.imag(), pi / 2));
        }
        else if (std::isfinite(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && std::isfinite(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(testcases[i].real()) == std::signbit(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, r.imag() == 0);
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(testcases[i].imag()) == std::signbit(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && std::isinf(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(testcases[i].real()) == std::signbit(r.real()));
            if (std::signbit(testcases[i].imag()))
                CALL_CHECK_OBJ_I(check_obj, i, is_about(r.imag(), -pi / 4));
            else
                CALL_CHECK_OBJ_I(check_obj, i, is_about(r.imag(), pi / 4));
        }
        else if (std::isinf(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
#if !_PSTL_TEST_COMPLEX_ASINH_BROKEN
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(testcases[i].real()) == std::signbit(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
#endif // _PSTL_TEST_COMPLEX_ASINH_BROKEN
        }
        else if (std::isnan(testcases[i].real()) && testcases[i].imag() == 0)
        {
#if !_PSTL_TEST_COMPLEX_ASINH_BROKEN
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, r.imag() == 0);
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(testcases[i].imag()) == std::signbit(r.imag()));
#endif // _PSTL_TEST_COMPLEX_ASINH_BROKEN
        }
        else if (std::isnan(testcases[i].real()) && std::isfinite(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
        }
        else if (std::isnan(testcases[i].real()) && std::isinf(testcases[i].imag()))
        {
#if !_PSTL_TEST_COMPLEX_ASINH_BROKEN
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
#endif // _PSTL_TEST_COMPLEX_ASINH_BROKEN
        }
        else if (std::isnan(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
        }
        else
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(r.real()) == std::signbit(testcases[i].real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
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
