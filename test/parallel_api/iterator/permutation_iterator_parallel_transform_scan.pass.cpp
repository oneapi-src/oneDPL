// -*- C++ -*-
//===-- permutation_iterator_parallel_transform_scan.pass.cpp -------------===//
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

#include "permutation_iterator_common.h"

// dpl::remove_if -> __parallel_transform_scan
// Requirements: only for random_access_iterator
DEFINE_TEST_PERM_IT(test_remove_if, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_remove_if)

    template <typename TIterator, typename Size>
    void generate_data(TIterator itBegin, TIterator itEnd, Size n)
    {
        Size index = 0;
        for (auto it = itBegin; it != itEnd; ++it, ++index)
            *it = (n - index) % 2 ? 0 : 1;
    }

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        if constexpr (is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>)
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // source data for remove_if
            const auto host_keys_ptr = host_keys.get();

            test_through_permutation_iterator<Iterator1, Size, PermItIndexTag>{first1, n}(
                [&](auto permItBegin, auto permItEnd)
                {
                    const auto testing_n = ::std::distance(permItBegin, permItEnd);

                    // Fill full source data set (not only values iterated by permutation iterator)
                    generate_data(host_keys_ptr, host_keys_ptr + n, n);
                    host_keys.update_data();

                    // Copy source data back
                    std::vector<TestValueType> sourceData(testing_n);
                    dpl::copy(exec, permItBegin, permItEnd, sourceData.begin());
                    wait_and_throw(exec);

                    const auto op = [](TestValueType val) { return val > 0; };

                    auto itEndNewRes = dpl::remove_if(exec, permItBegin, permItEnd, op);
                    wait_and_throw(exec);

                    const auto newSizeResult = ::std::distance(permItBegin, itEndNewRes);

                    // Copy modified data back
                    std::vector<TestValueType> resultRemoveIf(newSizeResult);
                    dpl::copy(exec, permItBegin, itEndNewRes, resultRemoveIf.begin());
                    wait_and_throw(exec);

                    // Eval expected result
                    auto expectedRemoveIf = sourceData;
                    auto itEndNewExpected = ::std::remove_if(expectedRemoveIf.begin(), expectedRemoveIf.end(), op);
                    const auto newSizeExpected = ::std::distance(expectedRemoveIf.begin(), itEndNewExpected);

                    // Check results
                    EXPECT_EQ(newSizeExpected, newSizeResult, "Wrong result size after dpl::remove_if");
                    EXPECT_EQ_N(expectedRemoveIf.begin(), resultRemoveIf.begin(), newSizeExpected, "Wrong result after dpl::remove_if");
                });
        }
    }
};

template <typename ValueType, typename PermItIndexTag>
void
run_algo_tests()
{
    constexpr ::std::size_t kZeroOffset = 0;

#if TEST_DPCPP_BACKEND_PRESENT
    // Run tests on <USM::shared, USM::device, sycl::buffer> + <all_hetero_policies>
    // dpl::remove_if -> __parallel_transform_scan (only for random_access_iterator)
    test1buffer<sycl::usm::alloc::shared, ValueType, test_remove_if<ValueType, PermItIndexTag>>();
    test1buffer<sycl::usm::alloc::device, ValueType, test_remove_if<ValueType, PermItIndexTag>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    // Run tests on <std::vector::iterator> + <all_host_policies>
    // dpl::remove_if -> __parallel_transform_scan (only for random_access_iterator)
    test_algo_one_sequence<ValueType, test_remove_if<ValueType, PermItIndexTag>>(kZeroOffset);
}

int
main()
{
    using ValueType = ::std::uint32_t;

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_tests<ValueType, perm_it_index_tags::usm_shared>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_tests<ValueType, perm_it_index_tags::counting>();
    run_algo_tests<ValueType, perm_it_index_tags::host>();
    run_algo_tests<ValueType, perm_it_index_tags::transform_iterator>();
    run_algo_tests<ValueType, perm_it_index_tags::callable_object>();

    return TestUtils::done();
}
