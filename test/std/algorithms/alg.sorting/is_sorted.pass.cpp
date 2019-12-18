// -*- C++ -*-
//===-- is_sorted.pass.cpp ------------------------------------------------===//
//
// Copyright (C) 2017-2019 Intel Corporation
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

#include "support/pstl_test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

using namespace TestUtils;

template <typename Type, std::size_t Partition>
struct test_is_sorted
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, bool exam)
    {
        using namespace std;
        typedef typename std::iterator_traits<Iterator>::value_type T;

        //try random-access iterator
        bool res = is_sorted(exec, first, last);
        EXPECT_TRUE(exam == res, "is_sorted wrong result for random-access iterator");
    }
};

template <typename Type, std::size_t Partition>
struct test_is_sorted_predicate
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, bool exam)
    {
        using namespace std;
        typedef typename std::iterator_traits<Iterator>::value_type T;

        //try random-access iterator with a predicate
        bool res = is_sorted(exec, first, last, std::less<T>());
        EXPECT_TRUE(exam == res, "is_sorted wrong result for random-access iterator");
    }
};

template <typename Type, std::size_t Partition>
struct test_is_sorted_until
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last)
    {
        using namespace std;
        typedef typename std::iterator_traits<Iterator>::value_type T;

        //try random-access iterator
        auto iexam = is_sorted_until(first, last);
        auto ires = is_sorted_until(exec, first, last);
        EXPECT_TRUE(iexam == ires, "is_sorted_until wrong result for random-access iterator");
    }
};

template <typename Type, std::size_t Partition>
struct test_is_sorted_until_predicate
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last)
    {
        using namespace std;
        typedef typename std::iterator_traits<Iterator>::value_type T;

        //try random-access iterator with a predicate
        auto iexam = is_sorted_until(first, last, std::less<T>());
        auto ires = is_sorted_until(exec, first, last, std::less<T>());
        EXPECT_TRUE(iexam == ires, "is_sorted_until wrong result for random-access iterator");
    }
};

template <typename T>
void
test_is_sorted_by_type()
{

    Sequence<T> in(99999, [](size_t v) -> T { return T(v); }); //fill 0..n

    invoke_on_all_policies(test_is_sorted<T, 0>(), in.begin(), in.end(), std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted_predicate<T, 0>(), in.begin(), in.end(),
                           std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted_until<T, 0>(), in.begin(), in.end());
    invoke_on_all_policies(test_is_sorted_until_predicate<T, 0>(), in.begin(), in.end());

    invoke_on_all_policies(test_is_sorted<T, 1>(), in.cbegin(), in.cend(), std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted_predicate<T, 1>(), in.cbegin(), in.cend(),
                           std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted_until<T, 1>(), in.cbegin(), in.cend());
    invoke_on_all_policies(test_is_sorted_until_predicate<T, 1>(), in.cbegin(), in.cend());

    in[in.size() / 2] = -1;
    invoke_on_all_policies(test_is_sorted<T, 2>(), in.begin(), in.end(), std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted_predicate<T, 2>(), in.begin(), in.end(),
                           std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted_until<T, 2>(), in.begin(), in.end());
    invoke_on_all_policies(test_is_sorted_until_predicate<T, 2>(), in.begin(), in.end());

    invoke_on_all_policies(test_is_sorted<T, 3>(), in.cbegin(), in.cend(), std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted_predicate<T, 3>(), in.cbegin(), in.cend(),
                           std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted_until<T, 3>(), in.cbegin(), in.cend());
    invoke_on_all_policies(test_is_sorted_until_predicate<T, 3>(), in.cbegin(), in.cend());

    in[1] = -1;
    invoke_on_all_policies(test_is_sorted<T, 4>(), in.begin(), in.end(), std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted_predicate<T, 4>(), in.begin(), in.end(),
                           std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted_until<T, 4>(), in.begin(), in.end());
    invoke_on_all_policies(test_is_sorted_until_predicate<T, 4>(), in.begin(), in.end());

    invoke_on_all_policies(test_is_sorted<T, 5>(), in.cbegin(), in.cend(), std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted_predicate<T, 5>(), in.cbegin(), in.cend(),
                           std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted_until<T, 5>(), in.cbegin(), in.cend());
    invoke_on_all_policies(test_is_sorted_until_predicate<T, 5>(), in.cbegin(), in.cend());

    //an empty container
    Sequence<T> in0(0);
    invoke_on_all_policies(test_is_sorted<T, 6>(), in0.begin(), in0.end(), std::is_sorted(in0.begin(), in0.end()));
    invoke_on_all_policies(test_is_sorted_predicate<T, 6>(), in0.begin(), in0.end(),
                           std::is_sorted(in0.begin(), in0.end()));
    invoke_on_all_policies(test_is_sorted_until<T, 6>(), in0.begin(), in0.end());
    invoke_on_all_policies(test_is_sorted_until_predicate<T, 6>(), in0.begin(), in0.end());

    invoke_on_all_policies(test_is_sorted<T, 7>(), in0.cbegin(), in0.cend(), std::is_sorted(in0.begin(), in0.end()));
    invoke_on_all_policies(test_is_sorted_predicate<T, 7>(), in0.cbegin(), in0.cend(),
                           std::is_sorted(in0.begin(), in0.end()));
    invoke_on_all_policies(test_is_sorted_until<T, 7>(), in0.cbegin(), in0.cend());
    invoke_on_all_policies(test_is_sorted_until_predicate<T, 7>(), in0.cbegin(), in0.cend());

    //non-descending order
    Sequence<T> in1(9, [](size_t v) -> T { return T(0); });
    invoke_on_all_policies(test_is_sorted<T, 8>(), in1.begin(), in1.end(), std::is_sorted(in1.begin(), in1.end()));
    invoke_on_all_policies(test_is_sorted_predicate<T, 8>(), in1.begin(), in1.end(),
                           std::is_sorted(in1.begin(), in1.end()));
    invoke_on_all_policies(test_is_sorted_until<T, 8>(), in1.begin(), in1.end());
    invoke_on_all_policies(test_is_sorted_until_predicate<T, 8>(), in1.begin(), in1.end());

    invoke_on_all_policies(test_is_sorted<T, 9>(), in1.cbegin(), in1.cend(), std::is_sorted(in1.begin(), in1.end()));
    invoke_on_all_policies(test_is_sorted_predicate<T, 9>(), in1.cbegin(), in1.cend(),
                           std::is_sorted(in1.begin(), in1.end()));
    invoke_on_all_policies(test_is_sorted_until<T, 9>(), in1.cbegin(), in1.cend());
    invoke_on_all_policies(test_is_sorted_until_predicate<T, 9>(), in1.cbegin(), in1.cend());
}

template <typename T>
struct test_non_const_is_sorted
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        is_sorted(exec, iter, iter, std::less<T>());
    }
};

template <typename T>
struct test_non_const_is_sorted_until
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        is_sorted_until(exec, iter, iter, std::less<T>());
    }
};

int
main()
{

    test_is_sorted_by_type<int32_t>();
    test_is_sorted_by_type<float64_t>();

    test_is_sorted_by_type<Wrapper<int32_t>>();

    test_algo_basic_single<int32_t>(run_for_rnd_fw<test_non_const_is_sorted<int32_t>>());
    test_algo_basic_single<int32_t>(run_for_rnd_fw<test_non_const_is_sorted_until<int32_t>>());

    std::cout << done() << std::endl;
    return 0;
}
