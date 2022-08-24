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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-constant-compare"

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

#pragma clang diagnostic pop
}

ONEDPL_TEST_NUM_MAIN
{
    test<float>();
    INVOKE_IF_DOUBLE_SUPPORT(test<double>())
    INVOKE_IF_LONG_DOUBLE_SUPPORT(test<long double>())
    INVOKE_IF_DOUBLE_SUPPORT(test_edges())

  return 0;
}
