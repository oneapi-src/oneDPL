// -*- C++ -*-
//===-- find_first_of.pass.cpp --------------------------------------------===//
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

template <typename T, std::size_t Partition>
struct test_find_first_of
{
#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                             \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Iterator2>
    void
    operator()(pstl::execution::unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub)
    {
    }

    template <typename Iterator1, typename Iterator2>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub)
    {
    }
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub)
    {
        using namespace std;
        Iterator1 expected = find_first_of(b, e, bsub, esub);
        Iterator1 actual = find_first_of(exec, b, e, bsub, esub);
        EXPECT_TRUE(actual == expected, "wrong return result from find_first_of");
    }
};

template <typename T, std::size_t Partition>
struct test_find_first_of_predicate
{
#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                             \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(pstl::execution::unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub,
               Predicate pred)
    {
    }

    template <typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(pstl::execution::parallel_unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub,
               Predicate pred)
    {
    }
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred)
    {
        using namespace std;
        Iterator1 expected = find_first_of(b, e, bsub, esub, pred);
        Iterator1 actual = find_first_of(exec, b, e, bsub, esub, pred);
        EXPECT_TRUE(actual == expected, "wrong return result from find_first_of with a predicate");
    }
};

template <typename T, typename Predicate>
void
test(Predicate pred)
{

    const std::size_t max_n1 = 1000;
    const std::size_t max_n2 = (max_n1 * 10) / 8;
    Sequence<T> in1(max_n1, [](std::size_t k) { return T(1); });
    Sequence<T> in2(max_n2, [](std::size_t k) { return T(0); });
    for (std::size_t n1 = 0; n1 <= max_n1; n1 = n1 <= 16 ? n1 + 1 : size_t(3.1415 * n1))
    {
        std::size_t sub_n[] = {0, 1, n1 / 3, n1, (n1 * 10) / 8};
        for (const auto n2 : sub_n)
        {
            invoke_on_all_policies(test_find_first_of<T, 0>(), in1.begin(), in1.begin() + n1, in2.data(),
                                   in2.data() + n2);
            invoke_on_all_policies(test_find_first_of_predicate<T, 0>(), in1.begin(), in1.begin() + n1, in2.data(),
                                   in2.data() + n2, pred);

            in2[n2 / 2] = T(1);
            invoke_on_all_policies(test_find_first_of<T, 1>(), in1.cbegin(), in1.cbegin() + n1, in2.data(),
                                   in2.data() + n2);
            invoke_on_all_policies(test_find_first_of_predicate<T, 1>(), in1.cbegin(), in1.cbegin() + n1, in2.data(),
                                   in2.data() + n2, pred);

            if (n2 >= 3)
            {
                in2[2 * n2 / 3] = T(1);
                invoke_on_all_policies(test_find_first_of<T, 2>(), in1.cbegin(), in1.cbegin() + n1, in2.begin(),
                                       in2.begin() + n2);
                invoke_on_all_policies(test_find_first_of_predicate<T, 2>(), in1.cbegin(), in1.cbegin() + n1,
                                       in2.begin(), in2.begin() + n2, pred);
                in2[2 * n2 / 3] = T(0);
            }
            in2[n2 / 2] = T(0);
        }
    }
    invoke_on_all_policies(test_find_first_of<T, 3>(), in1.begin(), in1.begin() + max_n1 / 10, in1.data(),
                           in1.data() + max_n1 / 10);
    invoke_on_all_policies(test_find_first_of_predicate<T, 3>(), in1.begin(), in1.begin() + max_n1 / 10, in1.data(),
                           in1.data() + max_n1 / 10, pred);
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename FirstIterator, typename SecondInterator>
    void
    operator()(Policy&& exec, FirstIterator first_iter, SecondInterator second_iter)
    {
        invoke_if(exec, [&]() {
            find_first_of(exec, first_iter, first_iter, second_iter, second_iter, non_const(std::equal_to<T>()));
        });
    }
};

int
main()
{
    test<int32_t>(std::equal_to<int32_t>());
    test<uint16_t>(std::not_equal_to<uint16_t>());
    test<float64_t>([](const float64_t x, const float64_t y) { return x * x == y * y; });

    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const<int32_t>>());

    std::cout << done() << std::endl;
    return 0;
}
