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

// <functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<const T> cref(reference_wrapper<T> t);

#include "support/test_config.h"

#include <oneapi/dpl/functional>

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
class KernelCRef2PassTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelCRef2PassTest>([=]() {
            const int i = 0;
            dpl::reference_wrapper<const int> r1 = std::cref(i);
            dpl::reference_wrapper<const int> r2 = std::cref(r1);
            ret_access[0] = (r2.get() == i);
        });
    });

    auto ret_access_host = buffer1.get_access<sycl::access::mode::read>();
    EXPECT_TRUE(ret_access_host[0], "Error in work with const reference");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
