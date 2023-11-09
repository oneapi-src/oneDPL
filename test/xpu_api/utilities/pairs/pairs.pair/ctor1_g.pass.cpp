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
struct X
{
    explicit X(int, int) {}

  private:
    X(const X&) = delete;
};

struct move_only
{
    move_only() {}
    move_only(move_only&&) {}

  private:
    move_only(const move_only&) = delete;
};

bool
kernel_test()
{
    int* ip = 0;
    int X::*mp = 0;

    dpl::pair<int*, int*> p1(0, 0);
    dpl::pair<int*, int*> p2(ip, 0);
    dpl::pair<int*, int*> p3(0, ip);
    dpl::pair<int*, int*> p4(ip, ip);

    dpl::pair<int X::*, int*> p5(0, 0);
    dpl::pair<int X::*, int X::*> p6(mp, 0);
    dpl::pair<int X::*, int X::*> p7(0, mp);
    dpl::pair<int X::*, int X::*> p8(mp, mp);
    dpl::pair<int*, move_only> p9(ip, move_only());
    dpl::pair<int X::*, move_only> p10(mp, move_only());
    dpl::pair<move_only, int*> p11(move_only(), ip);
    dpl::pair<move_only, int X::*> p12(move_only(), mp);
    dpl::pair<move_only, move_only> p13{move_only(), move_only()};
    return true;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() { ret_access[0] = kernel_test(); });
        });
    }
    EXPECT_TRUE(ret, "Wrong result of dpl::pair constructors check in kernel_test");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
