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

// Tuple

#include "support/test_config.h"

#include <oneapi/dpl/tuple>
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
// A simple class without conversions to check some things
struct foo
{
};

bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItem{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItem);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                using dpl::ignore;
                //test construction
                typedef dpl::tuple<int, int, int, int, int, int, int, int, int, int> type1;
                type1 a(0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
                type1 b(0, 0, 0, 0, 0, 0, 0, 0, 0, 2);
                type1 c(a);
                typedef dpl::tuple<int, int, int, int, int, int, int, int, int, char> type2;
                type2 d(0, 0, 0, 0, 0, 0, 0, 0, 0, 3);
                type1 e(d);
                typedef dpl::tuple<foo, int, int, int, int, int, int, int, int, foo> type3;
                // get
                ret_access[0] &= (dpl::get<9>(a) == 1 && dpl::get<9>(b) == 2);
                // comparisons
                ret_access[0] &= (a == a && !(a != a) && a <= a && a >= a && !(a < a) && !(a > a));
                ret_access[0] &= (!(a == b) && a != b && a <= b && a < b && !(a >= b) && !(a > b));
                //tie
                {
                    int i = 0;
                    tie(ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore, ignore, i) = a;
                    ret_access[0] &= (i == 1);
                }
                //test_assignment
                a = d;
                a = b;
                //make_tuple
                dpl::make_tuple(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

                //dpl::tuple_size
                ret_access[0] &= (dpl::tuple_size<type3>::value == 10);
                //dpl::tuple_element
                {
                    foo q1;
                    dpl::tuple_element<0, type3>::type q2(q1);
                    dpl::tuple_element<9, type3>::type q3(q1);
                }
            });
        });
    }
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main(void)
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of big tuples check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
