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
    dpl::complex<T> a = dpl::sqrt(c);
    assert(is_about(dpl::real(a), dpl::real(x)));
    assert(std::abs(dpl::imag(c)) < T(1.e-6));
}

template <class T>
void
test()
{
    test(dpl::complex<T>(64, 0), dpl::complex<T>(8, 0));
}

template <class TChecker>
void test_edges(TChecker& check_obj)
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::sqrt(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            CALL_CHECK_OBJ_I(check_obj, i, !std::signbit(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
        else if (std::isinf(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, r.real() > 0);
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.imag()));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(r.imag()) == std::signbit(testcases[i].imag()));
        }
        else if (std::isfinite(testcases[i].real()) && std::isnan(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isfinite(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, r.real() == 0);
            CALL_CHECK_OBJ_I(check_obj, i, !std::signbit(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.imag()));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(testcases[i].imag()) == std::signbit(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isfinite(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, r.real() > 0);
            CALL_CHECK_OBJ_I(check_obj, i, r.imag() == 0);
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(testcases[i].imag()) == std::signbit(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 && std::isnan(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.imag()));
        }
        else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 && std::isnan(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, r.real() > 0);
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
        }
        else if (std::isnan(testcases[i].real()) && (std::isfinite(testcases[i].imag()) || std::isnan(testcases[i].imag())))
        {
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r.imag()));
        }
        else if (std::signbit(testcases[i].imag()))
        {
            CALL_CHECK_OBJ_I(check_obj, i, !std::signbit(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(r.imag()));
        }
        else
        {
            CALL_CHECK_OBJ_I(check_obj, i, !std::signbit(r.real()));
            CALL_CHECK_OBJ_I(check_obj, i, !std::signbit(r.imag()));
        }
    }
}

ONEDPL_TEST_NUM_MAIN
{
#if _PSTL_ICC_TEST_COMPLEX_MSVC_MATH_DOUBLE_REQ
    IF_DOUBLE_SUPPORT(test<float>())
#else
    test<float>();
#endif
    IF_DOUBLE_SUPPORT(test<double>())
    IF_LONG_DOUBLE_SUPPORT(test<long double>())
    IF_DOUBLE_SUPPORT_REF_CAPT(test_edges(check_obj))

  return 0;
}
