//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   T
//   abs(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test()
{
    dpl::complex<T> z(3, 4);
    assert(dpl::abs(z) == 5);
}

template <class TChecker>
void test_edges(TChecker& check_obj)
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        double r = dpl::abs(testcases[i]);
        switch (classify(testcases[i]))
        {
        case zero:
            CALL_CHECK_OBJ_I(check_obj, i, r == 0);
            CALL_CHECK_OBJ_I(check_obj, i, !std::signbit(r));
            break;
        case non_zero:
            CALL_CHECK_OBJ_I(check_obj, i, std::isfinite(r) && r > 0);
            break;
        case inf:
            CALL_CHECK_OBJ_I(check_obj, i, std::isinf(r) && r > 0);
            break;
        case NaN:
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r));
            break;
        case non_zero_nan:
            CALL_CHECK_OBJ_I(check_obj, i, std::isnan(r));
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
