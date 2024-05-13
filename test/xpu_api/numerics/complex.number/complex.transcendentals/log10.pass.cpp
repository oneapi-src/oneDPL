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
//   log10(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test(const dpl::complex<T>& c, dpl::complex<T> x)
{
    assert(dpl::log10(c) == x);
}

template <class T>
void
test()
{
    test(dpl::complex<T>(0, 0), dpl::complex<T>(-INFINITY, 0));
}

template <class TChecker>
void test_edges(TChecker& check_obj)
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        dpl::complex<double> r = dpl::log10(testcases[i]);
        dpl::complex<double> z = dpl::log(testcases[i]) / std::log(10);
        if (std::isnan(dpl::real(r)))
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(dpl::real(z)));
        else
        {
#if !_PSTL_TEST_COMPLEX_LOG10_BROKEN
            CALL_CHECK_OBJ_I(check_obj, i, dpl::real(r) == dpl::real(z));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(dpl::real(r)) == std::signbit(dpl::real(z)));
#endif // _PSTL_TEST_COMPLEX_LOG10_BROKEN
        }
        if (std::isnan(dpl::imag(r)))
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(dpl::imag(z)));
        else
        {
#if !_PSTL_TEST_COMPLEX_LOG10_BROKEN
            CALL_CHECK_OBJ_I(check_obj, i, dpl::imag(r) == dpl::imag(z));
            CALL_CHECK_OBJ_I(check_obj, i, std::signbit(dpl::imag(r)) == std::signbit(dpl::imag(z)));
#endif // _PSTL_TEST_COMPLEX_LOG10_BROKEN
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
