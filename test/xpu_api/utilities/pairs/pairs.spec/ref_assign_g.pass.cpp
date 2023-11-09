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
kernel_test1()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                typedef dpl::pair<int&, int> pair_type;
                int i = 1;
                int j = 2;
                pair_type p(i, 3);
                const pair_type q(j, 4);
                p = q;
                ret_acc[0] = (p.first == q.first);
                ret_acc[0] &= (p.second == q.second);
                ret_acc[0] &= (i == j);
            });
        });
    }
    return ret;
}

bool
kernel_test2()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                typedef dpl::pair<int, int&> pair_type;
                int i = 1;
                int j = 2;
                pair_type p(3, i);
                const pair_type q(4, j);
                p = q;
                ret_acc[0] = (p.first == q.first);
                ret_acc[0] &= (p.second == q.second);
                ret_acc[0] &= (i == j);
            });
        });
    }
    return ret;
}
bool
kernel_test3()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest3>([=]() {
                typedef dpl::pair<int&, int&> pair_type;
                int i = 1;
                int j = 2;
                int k = 3;
                int l = 4;
                pair_type p(i, j);
                const pair_type q(k, l);
                p = q;
                ret_acc[0] = (p.first == q.first);
                ret_acc[0] &= (p.second == q.second);
                ret_acc[0] &= (i == k);
                ret_acc[0] &= (j == l);
            });
        });
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test1() && kernel_test2() && kernel_test3();
    EXPECT_TRUE(ret, "Wrong result of dpl::pair<T1&, T2&>::operator= check in kernel_test1, kernel_test2 or kernel_test3");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
