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
//   norm(const complex<T>& x);

#include "support/test_complex.h"

#include "../cases.h"

template <class T>
void
test()
{
    dpl::complex<T> z(3, 4);
    assert(norm(z) == 25);
}

void test_edges()
{
CLANG_DIAGNOSTIC_PUSH
CLANG_DIAGNOSTIC_IGNORED_AUTOLOGICAL_CONSTANT_COMPARE

    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        double r = dpl::norm(testcases[i]);
        switch (classify(testcases[i]))
        {
        case zero:
            assert(r == 0);
            assert(!std::signbit(r));
            break;
        case non_zero:
            assert(std::isfinite(r) && r > 0);
            break;
        case inf:
            assert(std::isinf(r) && r > 0);
            break;
        case NaN:
            assert(std::isnan(r));
            break;
        case non_zero_nan:
            assert(std::isnan(r));
            break;
        }
    }

CLANG_DIAGNOSTIC_POP
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    IF_DOUBLE_SUPPORT_IN_RUNTIME(test<double>())
    IF_CAN_COMPILE_LONG_DOUBLE(test<long double>())
    IF_DOUBLE_SUPPORT_IN_RUNTIME(test_edges())

  return 0;
}
