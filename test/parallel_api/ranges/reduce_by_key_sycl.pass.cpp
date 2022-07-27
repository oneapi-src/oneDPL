// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include "support/test_config.h"
#include "support/utils.h"

#include <iostream>
#include <vector>
#include <cassert>

#if TEST_DPCPP_BACKEND_PRESENT
#include <CL/sycl.hpp>
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
// Uncomment for extended logging
// #define _ONEDPL_DEBUG_SYCL 1
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto policy = oneapi::dpl::execution::dpcpp_default;

    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(policy.queue());

    // this line is fine -> [1], [6]
    std::vector<int, decltype(alloc)> keys  ({ 1, 1, 1 }, alloc);
    std::vector<int, decltype(alloc)> values({ 1, 2, 3 }, alloc);

    // this won't produce output -> expected [0], [6], but evaluated [0], [1]
    //std::vector<int, decltype(alloc)> keys  ({ 0, 0, 0 }, alloc);
    //std::vector<int, decltype(alloc)> values({ 1, 2, 3 }, alloc);

    std::vector<int, decltype(alloc)> output_keys(keys.size(), alloc);
    std::vector<int, decltype(alloc)> output_values(values.size(), alloc);

    // keys:                   [0,0,0]
    // values:                 [1,2,3]
    // expected output_keys:   [0]
    // expected output_values: [1+2+3=6]
    auto new_end = oneapi::dpl::reduce_by_key(policy, keys.begin(), keys.end(), values.begin(), output_keys.begin(),
                                              output_values.begin(), std::equal_to<int>(), std::plus<int>());

#if _ONEDPL_DEBUG_SYCL
    size_t size = oneapi::dpl::distance(output_keys.begin(), new_end.first);
    for (size_t i = 0; i < size; i++)
    {
        std::cout << "output_keys[" << i << "] = " << output_keys[i] << std::endl;
        std::cout << "output_values[" << i << "] = " << output_values[i] << std::endl;
    }
#endif // _ONEDPL_DEBUG_SYCL

    for (int key_val = -1; key_val < 2; ++key_val)
    {
        // Check on interval from 0 till 4096 * 3.5 (+1024)
        for (int destLength = 0; destLength <= 14336; destLength += (destLength < 10 ? 1 : (destLength < 100 ? 10 : (destLength < 1000 ? 100 : (destLength == 1000 ? 24 : 1024)))))
        {
#if _ONEDPL_DEBUG_SYCL
            std::cout << "key_val = " << key_val << ", destLength = " << destLength;
#endif // _ONEDPL_DEBUG_SYCL

            std::vector<int, decltype(alloc)> keys         (destLength, key_val, alloc);
            std::vector<int, decltype(alloc)> values       (destLength, 1,       alloc);
            std::vector<int, decltype(alloc)> output_keys  (destLength, alloc);
            std::vector<int, decltype(alloc)> output_values(destLength, alloc);

            auto new_end = oneapi::dpl::reduce_by_key(policy,
                                                      keys.begin(), keys.end(),
                                                      values.begin(), output_keys.begin(),
                                                      output_values.begin(),
                                                      std::equal_to<int>(),
                                                      std::plus<int>());

            const size_t size = oneapi::dpl::distance(output_keys.begin(), new_end.first);
            EXPECT_TRUE((destLength == 0 && size == 0) || size == 1, "wrong reduce_by_key result");

            for (size_t i = 0; i < size; i++)
            {
                EXPECT_EQ(key_val, output_keys[i], "wrong key");
                EXPECT_EQ(destLength, output_values[i], "wrong value");
            }

#if _ONEDPL_DEBUG_SYCL
            std::cout << " - OK" << std::endl;
#endif // _ONEDPL_DEBUG_SYCL
        }
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}