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
    bool check = false;
    sycl::range<1> numOfItem{1};
    dpl::pair<dpl::pair<int, int>, int> p;
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        sycl::buffer<bool, 1> buffer2(&check, numOfItem);
        sycl::buffer<decltype(p), 1> buffer3(&p, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto check_acc = buffer2.get_access<sycl::access::mode::write>(cgh);
            auto acc1 = buffer3.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                // check if there is change from input after data transfer
                check_acc[0] = (acc1[0].first == dpl::pair<int, int>());
                check_acc[0] &= (acc1[0].second == 0);
                if (check_acc[0])
                {
                    static_assert(sizeof(acc1[0]) == (3 * sizeof(int)), "assertion fail");
                    ret_acc[0] = ((void*)&acc1[0] == (void*)&acc1[0].first);
                    ret_acc[0] &= ((void*)&acc1[0] == (void*)&acc1[0].first.first);
                }
            });
        });
    }
    // check data after executing kernel function
    check &= (p.first == dpl::pair<int, int>());
    check &= (p.second == 0);
    if (!check)
        return false;
    return ret;
}
struct empty
{
};

bool
kernel_test2()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItem{1};
    dpl::pair<dpl::pair<empty, empty>, empty> p;

    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        sycl::buffer<decltype(p), 1> buffer2(&p, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto acc1 = buffer2.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                static_assert(sizeof(acc1[0]) == (3 * sizeof(empty)), "assertion fail");
                ret_acc[0] = ((void*)&acc1[0] == (void*)&acc1[0].first);
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
    typedef dpl::pair<int, int> int_pair;
    typedef dpl::pair<int_pair, int_pair> int_pair_pair;
    dpl::pair<int_pair_pair, int_pair_pair> p;

    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        sycl::buffer<decltype(p), 1> buffer2(&p, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            auto acc1 = buffer2.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest3>([=]() {
                static_assert(sizeof(int_pair_pair) == (2 * sizeof(int_pair)), "nested");
                static_assert(sizeof(acc1[0]) == (2 * sizeof(int_pair_pair)), "nested again");
                ret_acc[0] = ((void*)&acc1[0] == (void*)&acc1[0].first);
                ret_acc[0] &= ((void*)&acc1[0] == (void*)&acc1[0].first.first);
                ret_acc[0] &= ((void*)&acc1[0] == (void*)&acc1[0].first.first.first);
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
    EXPECT_TRUE(ret, "Wrong result of dpl::pair constructor check in kernel_test1, kernel_test2 or kernel_test3");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
