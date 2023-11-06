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
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

#if TEST_DPCPP_BACKEND_PRESENT
bool
kernel_test1(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                dpl::forward_as_tuple();

                ret_access[0] &= (dpl::get<0>(dpl::forward_as_tuple(-1)) == -1);
                ret_access[0] &= ((dpl::is_same<decltype(dpl::forward_as_tuple(-1)), dpl::tuple<int&&>>::value));

                const int i1 = 1;
                const int i2 = 2;
                const float d1 = 4.0f;
                auto t1 = dpl::forward_as_tuple(i1, i2, d1);
                ret_access[0] &= ((dpl::is_same<decltype(t1), dpl::tuple<const int&, const int&, const float&>>::value));
                ret_access[0] &= (dpl::get<0>(t1) == i1);
                ret_access[0] &= (dpl::get<1>(t1) == i2);
                ret_access[0] &= (dpl::get<2>(t1) == d1);

                typedef const int a_type1[3];
                a_type1 a1 = {-1, 1, 2};
                auto t2 = dpl::forward_as_tuple(a1);
                ret_access[0] &= ((dpl::is_same<decltype(t2), dpl::tuple<a_type1&>>::value));
                ret_access[0] &= (dpl::get<0>(t2)[0] == a1[0]);
                ret_access[0] &= (dpl::get<0>(t2)[1] == a1[1]);
                ret_access[0] &= (dpl::get<0>(t2)[2] == a1[2]);

                typedef int a_type2[2];
                a_type2 a2 = {2, -2};
                volatile int i4 = 1;
                auto t3 = dpl::forward_as_tuple(a2, i4);
                ret_access[0] &= ((dpl::is_same<decltype(t3), dpl::tuple<a_type2&, volatile int&>>::value));
                ret_access[0] &= (dpl::get<0>(t3)[0] == a2[0]);
                ret_access[0] &= (dpl::get<0>(t3)[1] == a2[1]);
                ret_access[0] &= (dpl::get<1>(t3) == i4);
            });
        });
    }
    return ret;
}

bool
kernel_test2(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                const int i1 = 1;
                const int i2 = 2;
                const double d1 = 4.0;
                auto t1 = dpl::forward_as_tuple(i1, i2, d1);
                ret_access[0] &= ((dpl::is_same<decltype(t1), dpl::tuple<const int&, const int&, const double&>>::value));
                ret_access[0] &= (dpl::get<0>(t1) == i1);
                ret_access[0] &= (dpl::get<1>(t1) == i2);
                ret_access[0] &= (dpl::get<2>(t1) == d1);
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
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    auto ret = kernel_test1(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        ret &= kernel_test2(deviceQueue);
    }
    EXPECT_TRUE(ret, "Wrong result of dpl::forward_as_tuple check in kernel_test1 or kernel_test2");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
