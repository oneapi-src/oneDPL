// -*- C++ -*-
//===-- transform_if.pass.cpp ----------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

// Tests for transform_if

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

using namespace TestUtils;

struct test_transform_if
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
              typename OutputIterator2, typename Size, typename UnaryOperation, typename Predicate>
    void
    operator()(Policy&& exec, InputIterator1 first, InputIterator1 last, InputIterator2 mask,
               InputIterator2 /* mask_end */, OutputIterator1 expected_first, OutputIterator1 expected_last,
               OutputIterator2 actual_first, OutputIterator2 actual_last, Size n, UnaryOperation op, Predicate pred)
    {
        // Try transform_if
        auto call1 = transform_if(exec, first, last, mask, expected_first, op, pred);
        auto call2 = oneapi::dpl::transform_if(exec, first, last, mask, actual_first, op, pred);
#if !TEST_DPCPP_BACKEND_PRESENT
        EXPECT_EQ_N(expected_first, actual_first, n, "Wrong effect from transform_if");
        for (size_t i = 0; i < GuardSize; ++i)
        {
            ++call2;
        }
        EXPECT_TRUE(actual_last == call2, "transform_if returned wrong iterator");

#else
        auto expected_count = ::std::distance(expected_first, call1);
        auto actual_count = ::std::distance(actual_first, call2);
        EXPECT_TRUE(expected_count == actual_count, "wrong return value from transform_if");

        EXPECT_EQ_N(expected_first, actual_first, expected_count, "Wrong effect from transform_if");

#endif
    }
};

template <typename In1, typename In2, typename Out>
void test() {
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
        Sequence<In1> in1(n, [](size_t k) { return (3 * k); });
        Sequence<In2> in2(n, [](size_t k) { return k % 2 == 0 ? 1 : 0; });

        Sequence<Out> out_expected(n, [](size_t) { return 0; });
        Sequence<Out> out_actual(n, [](size_t) { return 0; });

        invoke_on_all_policies<0>()(test_transform_if(), in1.begin(), in1.end(), in2.begin(), in2.end(),
                                    out_expected.begin(), out_expected.end(), out_actual.begin(), out_actual.end(), n,
                                    std::negate<int>(), oneapi::dpl::identity());
        invoke_on_all_policies<1>()(test_transform_if(), in1.cbegin(), in1.cend(), in2.cbegin(), in2.cend(),
                                    out_expected.begin(), out_expected.end(), out_actual.begin(), out_actual.end(), n,
                                    std::negate<int>(), oneapi::dpl::identity());
    }
}

int
main()
{
    test<int32_t, int32_t, int32_t>();

    return done();
}
