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
class KernelTupleLTTest1;
class KernelTupleLTTest2;

void
kernel_test1(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelTupleLTTest1>([=]() {
            {
                typedef dpl::tuple<> T1;
                typedef dpl::tuple<> T2;
                const T1 t1;
                const T2 t2;
                ret_access[0] = (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }

            {
                typedef dpl::tuple<long> T1;
                typedef dpl::tuple<float> T2;
                const T1 t1(1);
                const T2 t2(1.f);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long> T1;
                typedef dpl::tuple<float> T2;
                const T1 t1(1);
                const T2 t2(0.9f);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long> T1;
                typedef dpl::tuple<float> T2;
                const T1 t1(1);
                const T2 t2(1.1f);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int> T1;
                typedef dpl::tuple<float, long> T2;
                const T1 t1(1, 2);
                const T2 t2(1.f, 2);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int> T1;
                typedef dpl::tuple<float, long> T2;
                const T1 t1(1, 2);
                const T2 t2(0.9f, 2);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int> T1;
                typedef dpl::tuple<float, long> T2;
                const T1 t1(1, 2);
                const T2 t2(1.1f, 2);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int> T1;
                typedef dpl::tuple<float, long> T2;
                const T1 t1(1, 2);
                const T2 t2(1.f, 1);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int> T1;
                typedef dpl::tuple<float, long> T2;
                const T1 t1(1, 2);
                const T2 t2(1.f, 3);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, float> T1;
                typedef dpl::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.f, 2, 3);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, float> T1;
                typedef dpl::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(0.9f, 2, 3);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, float> T1;
                typedef dpl::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.1f, 2, 3);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, float> T1;
                typedef dpl::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.f, 1, 3);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, float> T1;
                typedef dpl::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.f, 3, 3);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, float> T1;
                typedef dpl::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.f, 2, 2);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, float> T1;
                typedef dpl::tuple<float, long, int> T2;
                const T1 t1(1, 2, 3.f);
                const T2 t2(1.f, 2, 4);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::tuple comparison operators check");
}

void
kernel_test2(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelTupleLTTest2>([=]() {
            {
                typedef dpl::tuple<> T1;
                typedef dpl::tuple<> T2;
                const T1 t1;
                const T2 t2;
                ret_access[0] = (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }

            {
                typedef dpl::tuple<long> T1;
                typedef dpl::tuple<double> T2;
                const T1 t1(1);
                const T2 t2(1);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long> T1;
                typedef dpl::tuple<double> T2;
                const T1 t1(1);
                const T2 t2(0.9);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long> T1;
                typedef dpl::tuple<double> T2;
                const T1 t1(1);
                const T2 t2(1.1);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int> T1;
                typedef dpl::tuple<double, long> T2;
                const T1 t1(1, 2);
                const T2 t2(1, 2);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int> T1;
                typedef dpl::tuple<double, long> T2;
                const T1 t1(1, 2);
                const T2 t2(0.9, 2);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int> T1;
                typedef dpl::tuple<double, long> T2;
                const T1 t1(1, 2);
                const T2 t2(1.1, 2);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int> T1;
                typedef dpl::tuple<double, long> T2;
                const T1 t1(1, 2);
                const T2 t2(1, 1);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int> T1;
                typedef dpl::tuple<double, long> T2;
                const T1 t1(1, 2);
                const T2 t2(1, 3);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, double> T1;
                typedef dpl::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1, 2, 3);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, double> T1;
                typedef dpl::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(0.9, 2, 3);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, double> T1;
                typedef dpl::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1.1, 2, 3);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, double> T1;
                typedef dpl::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1, 1, 3);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, double> T1;
                typedef dpl::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1, 3, 3);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, double> T1;
                typedef dpl::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1, 2, 2);
                ret_access[0] &= (!(t1 < t2));
                ret_access[0] &= (!(t1 <= t2));
                ret_access[0] &= ((t1 > t2));
                ret_access[0] &= ((t1 >= t2));
            }
            {
                typedef dpl::tuple<long, int, double> T1;
                typedef dpl::tuple<double, long, int> T2;
                const T1 t1(1, 2, 3);
                const T2 t2(1, 2, 4);
                ret_access[0] &= ((t1 < t2));
                ret_access[0] &= ((t1 <= t2));
                ret_access[0] &= (!(t1 > t2));
                ret_access[0] &= (!(t1 >= t2));
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::tuple comparison check");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    kernel_test1(deviceQueue);
    if (TestUtils::has_type_support<double>(deviceQueue.get_device()))
    {
        kernel_test2(deviceQueue);
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
