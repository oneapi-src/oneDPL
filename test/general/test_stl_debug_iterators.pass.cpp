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

#include <algorithm>
#include <memory>

#include "support/utils.h"

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    const int n = 100;
    sycl::queue q = TestUtils::get_test_queue();
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    auto usm_deleter = [q](int* ptr) { sycl::free(ptr, q); };
    std::unique_ptr<int, decltype(usm_deleter)> usm_uptr_1(sycl::malloc_shared<int>(n, q), usm_deleter);
    int* usm_ptr_1 = usm_uptr_1.get();

    auto event1 = q.fill(usm_ptr_1, 42, n);

    auto event_2 = q.parallel_for<class Test1>(sycl::range<>(n), {event1}, [=](auto id) {
        auto res1 = std::upper_bound(usm_ptr_1, usm_ptr_1 + n, 41) - usm_ptr_1;
        auto res2 = std::upper_bound(usm_ptr_1, usm_ptr_1 + n, 42) - usm_ptr_1;
    });
    event_2.wait();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
