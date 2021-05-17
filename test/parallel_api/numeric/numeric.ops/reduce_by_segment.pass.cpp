// -*- C++ -*-
//===-- reduce_by_segment.pass.cpp --------------------------------------------===//
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

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/numeric"
#include "oneapi/dpl/iterator"

#include "support/test_config.h"
#include "support/utils.h"

#include <iostream>
#include <iomanip>

#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/utils_sycl.h"

#    include <CL/sycl.hpp>

using namespace TestUtils;
using namespace oneapi::dpl::execution;

struct test_reduce_by_segment
{

    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Size>
    void
    initialize_data(Iterator1 keys_first, Iterator2 vals_first, Iterator3 key_res_first, Iterator4 val_res_first,
                    Size n)
    {
        //T keys[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, ..., 0 };
        //T vals[n1] = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, ..., 0 };

        auto host_keys = get_host_pointer(keys_first);
        auto host_vals = get_host_pointer(vals_first);
        auto host_key_res = get_host_pointer(key_res_first);
        auto host_val_res = get_host_pointer(val_res_first);

        for (int i = 0; i<n - 1; i> 3 ? i += 2 : ++i)
        {
            host_keys[i] = i % 4 + 1;
            host_vals[i] = i % 4 + 1;
            host_key_res[i] = 9;
            host_val_res[i] = 1;
            if (i > 3)
            {
                host_keys[i + 1] = host_keys[i];
                host_vals[i + 1] = host_vals[i];
                host_key_res[i + 1] = 9;
                host_val_res[i + 1] = 1;
            }
        }
        host_keys[n - 1] = 0;
        host_vals[n - 1] = 0;
    }

    template <typename Iterator1, typename Iterator2, typename Size>
    void
    check_values(Iterator1 key_res, Iterator2 val_res, Size n)
    {
        // keys_result = {1, 2, 3, 4, 1, 3, 1, 3, ..., 0};
        // vals_result = {1, 2, 3, 4, 2, 6, 2, 6, ..., 0};

        auto host_key_res = get_host_pointer(key_res);
        auto host_val_res = get_host_pointer(val_res);

        for (auto i = 0; i != n; ++i)
        {
            if (i == n - 1)
            {
                EXPECT_TRUE(host_key_res[i] == 0, "wrong effect from reduce_by_segment");
                EXPECT_TRUE(host_val_res[i] == 0, "wrong effect from reduce_by_segment");
            }
            else if (i < 4)
            {
                EXPECT_TRUE(host_key_res[i] == i + 1, "wrong effect from reduce_by_segment");
                EXPECT_TRUE(host_val_res[i] == i + 1, "wrong effect from reduce_by_segment");
            }
            else if (i % 2 == 0)
            {
                EXPECT_TRUE(host_key_res[i] == 1, "wrong effect from reduce_by_segment");
                EXPECT_TRUE(host_val_res[i] == 2 || host_val_res[i] == 1, "wrong effect from reduce_by_segment");
            }
            else
            {
                EXPECT_TRUE(host_key_res[i] == 3, "wrong effect from reduce_by_segment");
                EXPECT_TRUE(host_val_res[i] == 6 || host_val_res[i] == 3, "wrong effect from reduce_by_segment");
            }
        }
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4,
              typename Size>
    void
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator3 key_res,
               Iterator4 val_res, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        // call algorithm with no optional arguments
        initialize_data(keys_first, vals_first, key_res, val_res, n);

        auto new_policy = make_new_policy<new_kernel_name<Policy, 0>>(exec);
        auto res1 = oneapi::dpl::reduce_by_segment(new_policy, keys_first, keys_last, vals_first, key_res, val_res);
#    if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#    endif
        Size result_size = std::distance(key_res, res1.first);
        check_values(key_res, val_res, result_size);

        // call algorithm with equality comparator
        initialize_data(keys_first, vals_first, key_res, val_res, n);

        auto new_policy2 = make_new_policy<new_kernel_name<Policy, 1>>(exec);
        auto res2 = oneapi::dpl::reduce_by_segment(new_policy2, keys_first, keys_last, vals_first, key_res, val_res,
                                                   ::std::equal_to<KeyT>());
#    if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#    endif
        result_size = std::distance(key_res, res2.first);
        check_values(key_res, val_res, result_size);

        // call algorithm with addition operator
        initialize_data(keys_first, vals_first, key_res, val_res, n);

        auto new_policy3 = make_new_policy<new_kernel_name<Policy, 2>>(exec);
        auto res3 = oneapi::dpl::reduce_by_segment(new_policy3, keys_first, keys_last, vals_first, key_res, val_res,
                                                   ::std::equal_to<KeyT>(), ::std::plus<ValT>());
#    if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#    endif
        result_size = std::distance(key_res, res3.first);
        check_values(key_res, val_res, result_size);
    }
};

template <typename T, typename TestName>
void
test4buffers()
{
    const sycl::queue& queue = my_queue; // usm requires queue
#    if _PSTL_SYCL_TEST_USM
    {
        // Allocate space for data using USM.
        auto sycl_deleter = [queue](T* mem) { sycl::free(mem, queue.get_context()); };
        ::std::unique_ptr<T, decltype(sycl_deleter)> key_head(
            (T*)sycl::malloc_shared(sizeof(T) * max_n, queue.get_device(), queue.get_context()), sycl_deleter);
        ::std::unique_ptr<T, decltype(sycl_deleter)> val_head(
            (T*)sycl::malloc_shared(sizeof(T) * max_n, queue.get_device(), queue.get_context()), sycl_deleter);
        ::std::unique_ptr<T, decltype(sycl_deleter)> key_res_head(
            (T*)sycl::malloc_shared(sizeof(T) * max_n, queue.get_device(), queue.get_context()), sycl_deleter);
        ::std::unique_ptr<T, decltype(sycl_deleter)> val_res_head(
            (T*)sycl::malloc_shared(sizeof(T) * max_n, queue.get_device(), queue.get_context()), sycl_deleter);

        T* keys = key_head.get();
        T* vals = key_head.get();
        T* key_res = key_res_head.get();
        T* val_res = val_res_head.get();

        for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
        {
            invoke_on_all_hetero_policies<0>()(test_reduce_by_segment(), keys, keys + n, vals, key_res, val_res, n);
        }
    }
#    endif

    // create buffers
    sycl::buffer<uint64_t, 1> key_buf{sycl::range<1>(max_n)};
    sycl::buffer<uint64_t, 1> val_buf{sycl::range<1>(max_n)};
    sycl::buffer<uint64_t, 1> key_res_buf{sycl::range<1>(max_n)};
    sycl::buffer<uint64_t, 1> val_res_buf{sycl::range<1>(max_n)};

    // create sycl iterators
    auto keys = oneapi::dpl::begin(key_buf);
    auto vals = oneapi::dpl::begin(val_buf);
    auto key_res = oneapi::dpl::begin(key_res_buf);
    auto val_res = oneapi::dpl::begin(val_res_buf);

    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        invoke_on_all_hetero_policies<1>()(test_reduce_by_segment(), keys, keys + n, vals, key_res, val_res, n);
    }
}
#endif

int
main()
{
    test4buffers<uint64_t, test_reduce_by_segment>();
    return TestUtils::done();
}
