// -*- C++ -*-
//===-- reverse.pass.cpp --------------------------------------------------===//
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

#include <iterator>

using namespace TestUtils;

template <typename T>
struct test_one_policy
{
#if _PSTL_ICC_18_VC141_TEST_SIMD_LAMBDA_RELEASE_BROKEN || _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN ||       \
    _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN // dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Iterator2>
    typename ::std::enable_if<is_same_iterator_category<Iterator1, ::std::random_access_iterator_tag>::value, void>::type
    operator()(oneapi::dpl::execution::unsequenced_policy, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b,
               Iterator2 actual_e)
    {
    }

    template <typename Iterator1, typename Iterator2>
    typename ::std::enable_if<is_same_iterator_category<Iterator1, ::std::random_access_iterator_tag>::value, void>::type
    operator()(oneapi::dpl::execution::parallel_unsequenced_policy, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b,
               Iterator2 actual_e)
    {
    }
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    typename ::std::enable_if<!is_same_iterator_category<Iterator1, ::std::forward_iterator_tag>::value>::type
    operator()(ExecutionPolicy&& exec, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b, Iterator2 actual_e)
    {
        using namespace std;

        copy(data_b, data_e, actual_b);

        reverse(exec, actual_b, actual_e);

        bool check = equal(data_b, data_e, reverse_iterator<Iterator2>(actual_e));

        EXPECT_TRUE(check, "wrong result of reverse");
    }

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    typename ::std::enable_if<is_same_iterator_category<Iterator1, ::std::forward_iterator_tag>::value>::type
    operator()(ExecutionPolicy&& exec, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b, Iterator2 actual_e)
    {
    }
};

template <typename T>
void
test()
{
    const ::std::size_t max_len = 100000;

    Sequence<T> actual(max_len);

    Sequence<T> data(max_len, [](::std::size_t i) { return T(i); });

    for (::std::size_t len = 0; len < max_len; len = len <= 16 ? len + 1 : ::std::size_t(3.1415 * len))
    {
        invoke_on_all_policies<>()(test_one_policy<T>(), data.begin(), data.begin() + len, actual.begin(),
                                   actual.begin() + len);
    }
}

template <typename T>
struct wrapper
{
    T t;
    wrapper() {}
    explicit wrapper(T t_) : t(t_) {}
    bool
    operator==(const wrapper<T>& a) const
    {
        return t == a.t;
    }
};

int
main()
{
    test<int32_t>();
    test<uint16_t>();
    test<float64_t>();

#if !_PSTL_ICC_17_TEST_MAC_RELEASE_32_BROKEN
    test<wrapper<float64_t>>();
#endif

    ::std::cout << done() << ::std::endl;
    return 0;
}
