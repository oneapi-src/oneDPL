// -*- C++ -*-
//===-- find_end.pass.cpp -------------------------------------------------===//
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

#include "support/pstl_test_config.h"
#include "support/utils.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#if  !defined(_PSTL_TEST_FIND_END) && !defined(_PSTL_TEST_SEARCH)
#define _PSTL_TEST_FIND_END
#define _PSTL_TEST_SEARCH
#endif

using namespace TestUtils;

template <typename T>
struct test_find_end
{
#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                             \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(oneapi::dpl::execution::unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub,
               Predicate pred)
    {
    }

    template <typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(oneapi::dpl::execution::parallel_unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub,
               Predicate pred)
    {
    }
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred)
    {
        using namespace std;
        auto expected = find_end(b, e, bsub, esub, pred);
        auto actual = find_end(exec, b, e, bsub, esub);
        EXPECT_TRUE(actual == expected, "wrong return result from find_end");
    }
};

template <typename T>
struct test_find_end_predicate
{
#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                             \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(oneapi::dpl::execution::unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub,
               Predicate pred)
    {
    }

    template <typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(oneapi::dpl::execution::parallel_unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub,
               Predicate pred)
    {
    }
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred)
    {
        using namespace std;
        auto expected = find_end(b, e, bsub, esub, pred);
        auto actual = find_end(exec, b, e, bsub, esub, pred);
        EXPECT_TRUE(actual == expected, "wrong return result from find_end with a predicate");
    }
};

template <typename T>
struct test_search
{
#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                             \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(oneapi::dpl::execution::unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub,
               Predicate pred)
    {
    }

    template <typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(oneapi::dpl::execution::parallel_unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub,
               Predicate pred)
    {
    }
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred)
    {
        using namespace std;
        auto expected = search(b, e, bsub, esub, pred);
        auto actual = search(exec, b, e, bsub, esub);
        EXPECT_TRUE(actual == expected, "wrong return result from search");
    }
};

template <typename T>
struct test_search_predicate
{
#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||                                                             \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN //dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(oneapi::dpl::execution::unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub,
               Predicate pred)
    {
    }

    template <typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(oneapi::dpl::execution::parallel_unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub,
               Predicate pred)
    {
    }
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred)
    {
        using namespace std;
        auto expected = search(b, e, bsub, esub, pred);
        auto actual = search(exec, b, e, bsub, esub, pred);
        EXPECT_TRUE(actual == expected, "wrong return result from search with a predicate");
    }
};

template <typename T>
void
test(const ::std::size_t bits)
{

    const ::std::size_t max_n1 = 1000;
    const ::std::size_t max_n2 = (max_n1 * 10) / 8;
    Sequence<T> in(max_n1, [max_n1, bits](::std::size_t k) { return T(2 * HashBits(max_n1, bits - 1) ^ 1); });
    Sequence<T> sub(max_n2, [max_n1, bits](::std::size_t k) { return T(2 * HashBits(max_n1, bits - 1)); });
    for (::std::size_t n1 = 0; n1 <= max_n1; n1 = n1 <= 16 ? n1 + 1 : size_t(3.1415 * n1))
    {
        ::std::size_t sub_n[] = {0, 1, 3, n1, (n1 * 10) / 8};
        ::std::size_t res[] = {0, 1, n1 / 2, n1};
        for (auto n2 : sub_n)
        {
            for (auto r : res)
            {
                ::std::size_t i = r, isub = 0;
                for (; i < n1 & isub < n2; ++i, ++isub)
                    in[i] = sub[isub];
#ifdef _PSTL_TEST_FIND_END
                invoke_on_all_policies<0>()(test_find_end<T>(), in.begin(), in.begin() + n1, sub.begin(),
                                            sub.begin() + n2, ::std::equal_to<T>());
                invoke_on_all_policies<1>()(test_find_end_predicate<T>(), in.begin(), in.begin() + n1, sub.begin(),
                                            sub.begin() + n2, ::std::equal_to<T>());
                invoke_on_all_policies<2>()(test_find_end<T>(), in.cbegin(), in.cbegin() + n1, sub.cbegin(),
                                            sub.cbegin() + n2, ::std::equal_to<T>());
                invoke_on_all_policies<3>()(test_find_end_predicate<T>(), in.cbegin(), in.cbegin() + n1, sub.cbegin(),
                                            sub.cbegin() + n2, ::std::equal_to<T>());
#endif
#ifdef _PSTL_TEST_SEARCH
                invoke_on_all_policies<4>()(test_search<T>(), in.begin(), in.begin() + n1, sub.begin(),
                                            sub.begin() + n2, ::std::equal_to<T>());
                invoke_on_all_policies<5>()(test_search_predicate<T>(), in.begin(), in.begin() + n1, sub.begin(),
                                            sub.begin() + n2, ::std::equal_to<T>());
                invoke_on_all_policies<6>()(test_search<T>(), in.cbegin(), in.cbegin() + n1, sub.cbegin(),
                                            sub.cbegin() + n2, ::std::equal_to<T>());
                invoke_on_all_policies<7>()(test_search_predicate<T>(), in.cbegin(), in.cbegin() + n1, sub.cbegin(),
                                            sub.cbegin() + n2, ::std::equal_to<T>());
#endif
            }
        }
    }
}

template <typename T>
struct test_non_const_find_end
{
    template <typename Policy, typename FirstIterator, typename SecondInterator>
    void
    operator()(Policy&& exec, FirstIterator first_iter, SecondInterator second_iter)
    {
        invoke_if(exec, [&]() {
            find_end(exec, first_iter, first_iter, second_iter, second_iter, non_const(::std::equal_to<T>()));
        });
    }
};

template <typename T>
struct test_non_const_search
{
    template <typename Policy, typename FirstIterator, typename SecondInterator>
    void
    operator()(Policy&& exec, FirstIterator first_iter, SecondInterator second_iter)
    {
        invoke_if(exec, [&]() {
            search(exec, first_iter, first_iter, second_iter, second_iter, non_const(::std::equal_to<T>()));
        });
    }
};

int
main()
{
    test<int32_t>(8 * sizeof(int32_t));
    test<uint16_t>(8 * sizeof(uint16_t));
    test<float64_t>(53);
#if !_PSTL_BACKEND_SYCL && !_PSTL_ICC_16_17_TEST_REDUCTION_BOOL_TYPE_RELEASE_64_BROKEN
    test<bool>(1);

#ifdef _PSTL_TEST_FIND_END
    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const_find_end<int32_t>>());
#endif
#ifdef _PSTL_TEST_SEARCH
    test_algo_basic_double<int32_t>(run_for_rnd_fw<test_non_const_search<int32_t>>());
#endif
#endif

    ::std::cout << done() << ::std::endl;
    return 0;
}
