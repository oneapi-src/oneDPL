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

#include <oneapi/dpl/functional>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

// dpl::not1 is removed since C++20
#if TEST_DPCPP_BACKEND_PRESENT && TEST_STD_VER == 17
class KernelNot1Test;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelNot1Test>([=]() {
            typedef dpl::logical_not<int> F;
            ret_access[0] = (dpl::not1(F())(36));
            ret_access[0] &= (!dpl::not1(F())(0));
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with dpl::logical_not");
}
#endif // TEST_DPCPP_BACKEND_PRESENT && TEST_STD_VER

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
#    if TEST_STD_VER == 17
    kernel_test();
#    endif // TEST_STD_VER
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && TEST_STD_VER == 17);
}
