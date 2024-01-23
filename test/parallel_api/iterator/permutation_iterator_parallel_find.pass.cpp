// -*- C++ -*-
//===-- permutation_iterator_parallel_find.pass.cpp ------------------------===//
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

// dpl::find, dpl::find_if, dpl::find_if_not -> __parallel_find -> _parallel_find_or
DEFINE_TEST_PERM_IT(test_find, PermItIndexTag, KernelName)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_find)

    template <typename TIterator>
    void generate_data(TIterator itBegin, TIterator itEnd, TestValueType initVal)
    {
        ::std::iota(itBegin, itEnd, initVal);
    }

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec_src, Iterator1 first1, Iterator1 last1, Size n)
    {
        assert(n > 0);

        if constexpr (is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>)
        {
            auto exec = create_new_policy<KernelName>(::std::forward<Policy>(exec_src));

            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // source data for find
            const auto host_keys_ptr = host_keys.get();

            // Fill full source data set
            generate_data(host_keys_ptr, host_keys_ptr + n, TestValueType{});
            host_keys.update_data();

            test_through_permutation_iterator<Iterator1, Size, PermItIndexTag>{first1, n}(
                [&](auto permItBegin, auto permItEnd)
                {
                    const auto testing_n = ::std::distance(permItBegin, permItEnd);

                    if (testing_n >= 2)
                    {
                        // Get value to find: the second value
                        TestValueType valueToFind{};
                        dpl::copy(exec, permItBegin + 1, permItBegin + 2, &valueToFind);
                        wait_and_throw(exec);

                        const auto result = dpl::find(exec, permItBegin, permItEnd, valueToFind);
                        wait_and_throw(exec);

                        EXPECT_TRUE(result != permItEnd, "Wrong result of dpl::find");

                        // Copy data back
                        TestValueType foundedVal{};
                        dpl::copy(exec, result, result + 1, &foundedVal);
                        wait_and_throw(exec);
                        EXPECT_EQ(foundedVal, valueToFind, "Incorrect value was found in dpl::find");
                    }
                });
        }
    }
};

template <typename ValueType, typename PermItIndexTag, typename KernelName>
void
run_algo_tests()
{
    constexpr ::std::size_t kZeroOffset = 0;

#if TEST_DPCPP_BACKEND_PRESENT
    // Run tests on <USM::shared, USM::device, sycl::buffer> + <all_hetero_policies>
    // dpl::find, dpl::find_if, dpl::find_if_not -> __parallel_find -> _parallel_find_or
    test1buffer<sycl::usm::alloc::shared, ValueType, test_find<ValueType, PermItIndexTag, new_kernel_name<KernelName, 10>>,
                                                     test_find<ValueType, PermItIndexTag, new_kernel_name<KernelName, 15>>>();
    test1buffer<sycl::usm::alloc::device, ValueType, test_find<ValueType, PermItIndexTag, new_kernel_name<KernelName, 20>>,
                                                     test_find<ValueType, PermItIndexTag, new_kernel_name<KernelName, 25>>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    // Run tests on <std::vector::iterator> + <all_host_policies>
    // dpl::find, dpl::find_if, dpl::find_if_not -> __parallel_find -> _parallel_find_or
    test_algo_one_sequence<ValueType, test_find<ValueType, PermItIndexTag, KernelName>>(kZeroOffset);
}

int
main()
{
    using ValueType = ::std::uint32_t;

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_tests<ValueType, perm_it_index_tags::usm_shared,         class KernelName1>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_tests<ValueType, perm_it_index_tags::counting,           class KernelName2>();
    run_algo_tests<ValueType, perm_it_index_tags::host,               class KernelName3>();
    run_algo_tests<ValueType, perm_it_index_tags::transform_iterator, class KernelName4>();
    run_algo_tests<ValueType, perm_it_index_tags::callable_object,    class KernelName5>();

    return TestUtils::done();
}
