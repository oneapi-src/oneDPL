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

#include _ONEAPI_STD_TEST_HEADER(algorithm)

#include <iostream>

#include "support/utils.h"
#include "testsuite_iterators.h"
#include "checkData.h"
#include "test_macros.h"

namespace test_ns = _ONEAPI_TEST_NAMESPACE;

#if TEST_DPCPP_BACKEND_PRESENT
constexpr auto sycl_write = sycl::access::mode::write;

using test_ns::upper_bound;

typedef test_container<int, forward_iterator_wrapper> Container;

bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    int array[] = {0, 0, 0, 0, 1, 1, 1, 1};
    auto tmp = array;
    const int N = sizeof(array) / sizeof(array[0]);
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
                int arr[] = {0, 0, 0, 0, 1, 1, 1, 1};
                // check if there is change after data transfer
                check_access[0] = check_data(&access[0], arr, N);
                auto ret = true;
                if (check_access[0])
                {
                    auto ret = true;
                    for (int i = 0; i < 5; ++i)
                    {
                        for (int j = 4; j < 7; ++j)
                        {
                            Container con(&access[0] + i, &access[0] + j);
                            ret &= (upper_bound(con.begin(), con.end(), 0).ptr == &access[0] + 4);
                        }
                    }
                    ret_access[0] = ret;
                }
            });
        }).wait();
    }
    // check if there is change after executing kernel function
    check = check_data(tmp, array, N);
    if (!check)
        return false;
    return ret;
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of upper_bound in kernel_test");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
