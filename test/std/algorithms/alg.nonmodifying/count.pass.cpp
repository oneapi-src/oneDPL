// -*- C++ -*-
//===-- count.pass.cpp ----------------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
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

// Tests for count and count_if
#include "support/pstl_test_config.h"
#include "support/utils.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#if  !defined(_PSTL_TEST_COUNT) && !defined(_PSTL_TEST_COUNT_IF)
#define _PSTL_TEST_COUNT
#define _PSTL_TEST_COUNT_IF
#endif

using namespace TestUtils;

template <typename Type>
struct test_count
{
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, T needle)
    {
        auto expected = ::std::count(first, last, needle);
        auto result = ::std::count(exec, first, last, needle);
        EXPECT_EQ(expected, result, "wrong count result");
    }
};

template <typename Type>
struct test_count_if
{
    template <typename Policy, typename Iterator, typename Predicate>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Predicate pred)
    {
        auto expected = ::std::count_if(first, last, pred);
        auto result = ::std::count_if(exec, first, last, pred);
        EXPECT_EQ(expected, result, "wrong count_if result");
    }
};

template <typename T>
class IsEqual
{
    T value;

  public:
    IsEqual(T value_, OddTag) : value(value_) {}
    bool
    operator()(const T& x) const
    {
        return x == value;
    }
};

template <typename In, typename T, typename Predicate, typename Convert>
void
test(T needle, Predicate pred, Convert convert)
{
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In> in(n, [=](size_t k) -> In {
            // Sprinkle "42" and "50" early, so that short sequences have non-zero count.
            return convert((n - k - 1) % 3 == 0 ? 42 : (n - k - 2) % 5 == 0 ? 50 : 3 * (int(k) % 1000 - 500));
        });
#ifdef _PSTL_TEST_COUNT
        invoke_on_all_policies<0>()(test_count<In>(), in.begin(), in.end(), needle);
        invoke_on_all_policies<1>()(test_count<In>(), in.cbegin(), in.cend(), needle);
#endif
#ifdef _PSTL_TEST_COUNT_IF
        invoke_on_all_policies<2>()(test_count_if<In>(), in.begin(), in.end(), pred);
        invoke_on_all_policies<3>()(test_count_if<In>(), in.cbegin(), in.cend(), pred);
#endif
    }
}

struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        auto is_even = [&](float64_t v) {
            uint32_t i = (uint32_t)v;
            return i % 2 == 0;
        };
        count_if(exec, iter, iter, non_const(is_even));
    }
};

int
main()
{
    test<int16_t>(42, IsEqual<int16_t>(50, OddTag()), [](int16_t j) { return j; });
#if !_PSTL_ICC_16_17_TEST_REDUCTION_RELEASE_BROKEN
    test<int32_t>(42, [](const int32_t& x) { return true; }, [](int32_t j) { return j; });
#endif
    test<float64_t>(42, IsEqual<float64_t>(50, OddTag()), [](int32_t j) { return float64_t(j); });
#if !_PSTL_BACKEND_SYCL
    test<Number>(Number(42, OddTag()), IsEqual<Number>(Number(50, OddTag()), OddTag()),
                 [](int32_t j) { return Number(j, OddTag()); });
#ifdef _PSTL_TEST_COUNT_IF
    test_algo_basic_single<int32_t>(run_for_rnd_fw<test_non_const>());
#endif
#endif

    ::std::cout << done() << ::std::endl;
    return 0;
}
