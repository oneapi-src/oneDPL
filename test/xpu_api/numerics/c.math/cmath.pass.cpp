//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <cmath>

#include "support/test_complex.h"

#include <oneapi/dpl/cmath>

// convertible to int/float/double/etc
template <class T, int N=0>
struct Value {
    operator T () { return T(N); }
};

// See PR21083
// Ambiguous is a user-defined type that defines its own overloads of cmath
// functions. When the std overloads are candidates too (by using or adl),
// they should not interfere.
struct Ambiguous : std::true_type { // ADL
    operator float () { return 0.f; }
    operator double () { return 0.; }
};

Ambiguous nearbyint(Ambiguous){ return Ambiguous(); }

template <typename HasDoubleSupportInRuntime, typename HasLongDoubleSupportInCompiletime>
void test_nearbyint()
{
    static_assert((std::is_same<decltype(dpl::nearbyint((float)0)), float>::value), "");
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((bool)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((unsigned short)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((unsigned int)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((unsigned long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((unsigned long long)0)), double>::value), ""))
    IF_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((double)0)), double>::value), ""))
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyint((long double)0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::nearbyintf(0)), float>::value), "");
    IF_LONG_DOUBLE_SUPPORT(static_assert((std::is_same<decltype(dpl::nearbyintl(0)), long double>::value), ""))
    static_assert((std::is_same<decltype(dpl::nearbyint(Ambiguous())), Ambiguous>::value), "");
    assert(dpl::nearbyint(1) == 1);
}

ONEDPL_TEST_NUM_MAIN
{
    ONEDPL_TEST_NUM_CALL(test_nearbyint)

    return 0;
}
