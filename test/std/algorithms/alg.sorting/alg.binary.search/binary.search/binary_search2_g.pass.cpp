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

#include <iostream>

#include "checkData.h"
#include "test_macros.h"

namespace test_ns = _ONEAPI_TEST_NAMESPACE;

#if TEST_DPCPP_BACKEND_PRESENT
constexpr auto sycl_write = sycl::access::mode::write;

// A comparison, equalivalent to std::greater<int> without the
// dependency on <functional>.
struct gt
{
    bool
    operator()(const int& x, const int& y) const
    {
        return x > y;
    }
};

// binary_search, with and without comparison predicate
bool
kernel_test()
{
    using test_ns::binary_search;
    sycl::queue deviceQueue = TestUtils::get_test_queue();

    const int A[] = {1, 2, 3, 3, 3, 5, 8};
    const int C[] = {8, 5, 3, 3, 3, 2, 1};
    auto A1 = A, C1 = C;
    const int N = sizeof(A) / sizeof(int);
    const int first = A[0];
    const int last = A[N - 1];
    bool ret = false;
    bool check = false; // for checking data transfer
    sycl::range<1> item1{1};
    sycl::range<1> itemN{N};

    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        sycl::buffer<int, 1> buffer2(A, itemN);
        sycl::buffer<int, 1> buffer3(C, itemN);
        sycl::buffer<bool, 1> buffer4(&check, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl_write>(cgh);
            auto access2 = buffer2.get_access<sycl_write>(cgh);
            auto access3 = buffer3.get_access<sycl_write>(cgh);
            auto check_access = buffer4.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                const int A1[] = {1, 2, 3, 3, 3, 5, 8};
                const int C1[] = {8, 5, 3, 3, 3, 2, 1};
                // check if there is change after data transfer
                check_access[0] = check_data(&access2[0], A1, N);
                check_access[0] &= check_data(&access3[0], C1, N);

                if (check_access[0])
                {
                    ret_access[0] = (binary_search(&access2[0], &access2[0] + N, 5));
                    ret_access[0] &= (binary_search(&access2[0], &access2[0] + N, first));
                    ret_access[0] &= (binary_search(&access2[0], &access2[0] + N, last));
                    ret_access[0] &= (!binary_search(&access2[0], &access2[0] + N, 4));

                    ret_access[0] &= (binary_search(&access3[0], &access3[0] + N, 5, gt()));
                    ret_access[0] &= (binary_search(&access3[0], &access3[0] + N, first, gt()));
                    ret_access[0] &= (binary_search(&access3[0], &access3[0] + N, last, gt()));
                    ret_access[0] &= (!binary_search(&access3[0], &access3[0] + N, 4, gt()));
                }
            });
        }).wait();
    }
    // check if there is change after executing kernel function
    check &= check_data(A, A1, N);
    check &= check_data(C, C1, N);
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
    EXPECT_TRUE(ret, "");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
