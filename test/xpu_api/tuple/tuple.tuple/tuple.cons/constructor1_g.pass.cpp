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
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
using namespace std;
bool
kernel_test1()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                int x1 = 0, x2 = 0;
                const int& z1 = x1;

                // Test empty constructor
                dpl::tuple<> ta __attribute__((unused));
                dpl::tuple<int, int> tb;
                // Test construction from values
                dpl::tuple<int, int> tc(x1, x2);
                dpl::tuple<int, int&> td(x1, x2);
                dpl::tuple<const int&> te(z1);
                x1 = 1;
                x2 = 1;
                ret_access[0] = (get<0>(td) == 0 && get<1>(td) == 1 && get<0>(te) == 1);

                // Test identical dpl::tuple copy constructor
                dpl::tuple<int, int> tf(tc);
                dpl::tuple<int, int> tg(td);
                dpl::tuple<const int&> th(te);
                // Test different dpl::tuple copy constructor
                dpl::tuple<int, float> ti(tc);
                dpl::tuple<int, float> tj(td);
                // dpl::tuple<int&, int&> tk(tc);
                dpl::tuple<const int&, const int&> tl(tc);
                dpl::tuple<const int&, const int&> tm(tl);
                // Test constructing from a pair
                pair<int, int> pair1(1, 1);
                const pair<int, int> pair2(pair1);
                dpl::tuple<int, int> tn(pair1);
                dpl::tuple<int, const int&> to(pair1);
                dpl::tuple<int, int> tp(pair2);
                dpl::tuple<int, const int&> tq(pair2);
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
    auto ret = kernel_test1();
    EXPECT_TRUE(ret, "Wrong result of dpl::tuple constructors check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
