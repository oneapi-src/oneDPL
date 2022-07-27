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

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto policy = oneapi::dpl::execution::dpcpp_default;

    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(policy.queue());

    for (int key_val = -1; key_val < 2; ++key_val)
    {
        // Check on interval from 0 till 4096 * 3.5 (+1024)
        for (int destLength = 0; destLength <= 14336; destLength += 100)
        {
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

            const size_t size = new_end.first - output_keys.begin();
            EXPECT_TRUE((destLength == 0 && size == 0) || size == 1, "wrong reduce_by_key result");

            for (size_t i = 0; i < size; i++)
            {
                EXPECT_EQ(key_val, output_keys[i], "wrong key");
                EXPECT_EQ(destLength, output_values[i], "wrong value");
            }
        }
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}