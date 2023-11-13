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
struct foo
{
};

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
            cgh.single_task<class KernelTest>([=]() {
                foo q1;
                dpl::tuple_element<0, dpl::tuple<foo, void, int>>::type q2(q1);
                dpl::tuple_element<2, dpl::tuple<void, int, foo>>::type q3(q1);
                dpl::tuple_element<0, const dpl::tuple<foo, void, int>>::type q4(q1);
                dpl::tuple_element<2, const dpl::tuple<void, int, foo>>::type q5(q1);
                dpl::tuple_element<0, volatile dpl::tuple<foo, void, int>>::type q6(q1);
                dpl::tuple_element<2, volatile dpl::tuple<void, int, foo>>::type q7(q1);
                dpl::tuple_element<0, const volatile dpl::tuple<foo, void, int>>::type q8(q1);
                dpl::tuple_element<2, const volatile dpl::tuple<void, int, foo>>::type q9(q1);
                ret_access[0] = true;
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of dpl::tuple_element check");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
