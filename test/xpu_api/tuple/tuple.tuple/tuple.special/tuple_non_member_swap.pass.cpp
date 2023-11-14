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

#include <oneapi/dpl/tuple>

#include "support/utils.h"
#include "support/move_only.h"

#if TEST_DPCPP_BACKEND_PRESENT
class KernelNonMemberSwapTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelNonMemberSwapTest>([=]() {
            {
                typedef dpl::tuple<> T;
                T t0;
                T t1;
                swap(t0, t1);
            }

            {
                typedef dpl::tuple<MoveOnly> T;
                T t0(MoveOnly(0));
                T t1(MoveOnly(1));
                swap(t0, t1);
                ret_access[0] = (dpl::get<0>(t0) == 1 && dpl::get<0>(t1) == 0);
            }

            {
                typedef dpl::tuple<MoveOnly, MoveOnly> T;
                T t0(MoveOnly(0), MoveOnly(1));
                T t1(MoveOnly(2), MoveOnly(3));
                swap(t0, t1);
                ret_access[0] &= (dpl::get<0>(t0) == 2 && dpl::get<1>(t0) == 3 && dpl::get<0>(t1) == 0 && dpl::get<1>(t1) == 1);
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::swap check");
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
