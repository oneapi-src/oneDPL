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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

using namespace TestUtils;

struct test_transform_if
{
    template <typename It1, typename It2, typename Ot, typename Size>
    bool
    check(It1 first, It1 last, It2 mask, Ot result, Size n)
    {
        int i = 0;
        int j = n - 1;

        for (; first != last; ++first, ++mask, ++result, ++i, --j)
        {
            if (n % 2 == 0)
            { // even # of elements in output sequence
                if (*mask == 1)
                {
                    if (i % 2 == 0 && *result != -(3 * i))
                    { // forward iterator case
                        return false;
                    }
                    else if (i % 2 == 1 && *result != -(3 * j))
                    { // reverse iterator case
                        return false;
                    }
                }
                else if (*mask == 0 && *result != 0)
                {
                    return false;
                }
            }
            else
            { // odd # of elements in output sequence
                if (*mask == 1)
                {
                    if (i % 2 == 0 && (*result != -(3 * i) && *result != -(3 * j)))
                    {
                        return false;
                    }
                }
                else if (*mask == 0 && *result != 0)
                {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Size>
    void
    operator()(Policy&& exec, InputIterator1 first, InputIterator1 last, InputIterator2 mask,
               InputIterator2 /* mask_end */, OutputIterator result, OutputIterator result_end, Size n)
    {

        // call transform_if
        transform_if(exec, first, last, mask, result, std::negate<int>(), oneapi::dpl::identity());

        EXPECT_TRUE(check(first, last, mask, result, n), "transform_if wrong result");
        // reset output elements to 0
        ::std::fill(result, result_end, 0);
    }
};

template <typename In1, typename In2, typename Out>
void
test()
{
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In1> in1(n, [=](size_t k) { return (3 * k); });
        Sequence<In2> in2(n, [=](size_t k) { return k % 2 == 0 ? 1 : 0; });

        Sequence<Out> out(n, [=](size_t) { return 0; });

        invoke_on_all_policies<0>()(test_transform_if(), in1.begin(), in1.end(), in2.begin(), in2.end(), out.begin(),
                                    out.end(), n);
        invoke_on_all_policies<1>()(test_transform_if(), in1.cbegin(), in1.cend(), in2.cbegin(), in2.cend(),
                                    out.begin(), out.end(), n);
    }
}

int
main()
{
    test<int32_t, int32_t, int32_t>();

    return done();
}
