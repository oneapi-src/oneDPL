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

#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    bool check = false;
    sycl::range<1> numOfItem{1};
    dpl::pair<int, int> a(1, 1), b(2, 2);
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        sycl::buffer<bool, 1> buffer2(&check, numOfItem);
        sycl::buffer<decltype(a), 1> buffer3(&a, numOfItem);
        sycl::buffer<decltype(b), 1> buffer4(&b, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto check_acc = buffer2.get_access<sycl::access::mode::write>(cgh);
            auto acc1 = buffer3.get_access<sycl::access::mode::write>(cgh);
            auto acc2 = buffer4.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                // check if there is change from input after data transfer
                check_acc[0] = (acc1[0].first == 1);
                check_acc[0] &= (acc1[0].second == 1);
                check_acc[0] &= (acc2[0].first == 2);
                check_acc[0] &= (acc2[0].second == 2);
                if (check_acc[0])
                {
                    acc1[0] = dpl::move(acc2[0]);
                    dpl::pair<int, int> c(dpl::move(acc1[0]));
                    ret_acc[0] = (c.first == 2 && c.second == 2);
                    ret_acc[0] &= (acc1[0].first == 2 && acc1[0].second == 2);
                }
            });
        });
    }
    // check data after executing kernel functio
    check &= (a.first == 2 && a.second == 2);
    check &= (b.first == 2 && b.second == 2);
    if (!check)
        return false;
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::pair move check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
