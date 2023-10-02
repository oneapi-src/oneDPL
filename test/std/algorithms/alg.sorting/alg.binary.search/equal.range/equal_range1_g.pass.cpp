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

#include "oneapi_std_test_config.h"

#include _ONEAPI_STD_TEST_HEADER(algorithm)
#include _ONEAPI_STD_TEST_HEADER(utility)

#include <iostream>

#include "testsuite_iterators.h"
#include "checkData.h"

namespace test_ns = _ONEAPI_TEST_NAMESPACE;

#if TEST_DPCPP_BACKEND_PRESENT
constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

using test_ns::equal_range;

typedef test_container<int, forward_iterator_wrapper> Container;

bool
kernel_test1()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    int array[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    const int N = sizeof(array) / sizeof(array[0]);
    auto tmp = array;
    bool ret = false;
    bool check = false;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<bool, 1> buffer2(&check, item1);
        sycl::buffer<int, 1> buffer3(array, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto check_access = buffer2.get_access<sycl_write>(cgh);
            auto access = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                auto ret = true;
                int arr[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
                // check if there is change after data transfer
                check_access[0] = check_data(access.get_pointer().get(), arr, N);
                if (check_access[0])
                {
                    for (int i = 0; i < 6; ++i)
                    {
                        for (int j = 6; j < 12; ++j)
                        {
                            Container con(access.get_pointer().get() + i, access.get_pointer().get() + j);
                            ret &= (equal_range(con.begin(), con.end(), 1).first.ptr == access.get_pointer().get() + std::max(i, 4));
                            ret &= (equal_range(con.begin(), con.end(), 1).second.ptr == access.get_pointer().get() + std::min(j, 8));
                        }
                    }
                    ret_access[0] = ret;
                }
            });
        }).wait();
    }
    // check if there is change after executing kernel function
    check &= check_data(tmp, array, N);
    if (!check)
        return false;
    return ret;
}

bool
kernel_test2()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    int array[] = {0, 0, 2, 2, 2};
    const int N = sizeof(array) / sizeof(array[0]);
    auto tmp = array;
    bool ret = false;
    bool check = false;
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<bool, 1> buffer2(&check, item1);
        sycl::buffer<int, 1> buffer3(array, itemN);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto check_access = buffer2.get_access<sycl_write>(cgh);
            auto access = buffer3.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest2>([=]() {
                int arr[] = {0, 0, 2, 2, 2};
                // check if there is change after data transfer
                check_access[0] = check_data(access.get_pointer().get(), arr, N);
                if (check_access[0])
                {
                    Container con(access.get_pointer().get(), access.get_pointer().get() + 5);
                    ret_access[0] = (equal_range(con.begin(), con.end(), 1).first.ptr == access.get_pointer().get() + 2);
                    ret_access[0] &= (equal_range(con.begin(), con.end(), 1).second.ptr == access.get_pointer().get() + 2);
                }
            });
        }).wait();
    }
    // check if there is change after executing kernel function
    check &= check_data(tmp, array, N);
    if (!check)
        return false;
    return ret;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test1();
    ret &= kernel_test2();
    EXPECT_TRUE(ret, "");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
