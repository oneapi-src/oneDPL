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

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                int j = 1;
                const int k = 2;
                dpl::tuple<int, int&, const int&> a(0, j, k);
                const dpl::tuple<int, int&, const int&> b(1, j, k);
                ret_access[0] = (dpl::get<0>(a) == 0 && dpl::get<1>(a) == 1 && dpl::get<2>(a) == 2);
                dpl::get<0>(a) = 3;
                dpl::get<1>(a) = 4;
                ret_access[0] &= (dpl::get<0>(a) == 3 && dpl::get<1>(a) == 4);
                ret_access[0] &= (j == 4);
                dpl::get<1>(b) = 5;
                ret_access[0] &= (dpl::get<0>(b) == 1 && dpl::get<1>(b) == 5 && dpl::get<2>(b) == 2);
                ret_access[0] &= (j == 5);
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
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::get check in kernel_test");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
