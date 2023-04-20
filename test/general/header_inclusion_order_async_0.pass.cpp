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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#if TEST_DPCPP_BACKEND_PRESENT
#    include _PSTL_TEST_HEADER(async)
#endif // TEST_DPCPP_BACKEND_PRESENT

#include <vector>

#include "support/utils.h"

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    sycl::queue q = TestUtils::get_test_queue();

    constexpr std::size_t n = 100;

    using T = float;
    using allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

    allocator alloc(q);
    std::vector<T, allocator> data(n, 1, alloc);

    auto f = oneapi::dpl::experimental::reduce_async(TestUtils::make_device_policy(q), data.begin(), data.end(), T(0),
                                                     std::plus());
    f.wait();

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done();
}
