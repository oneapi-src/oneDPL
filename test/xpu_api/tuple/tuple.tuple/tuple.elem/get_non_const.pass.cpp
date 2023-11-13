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
struct Empty
{
};

struct S
{
    dpl::tuple<int, Empty> a;
    int k;
    Empty e;
    constexpr S() : a{1, Empty{}}, k(dpl::get<0>(a)), e(dpl::get<1>(a)) {}
};

constexpr dpl::tuple<int, int>
getP()
{
    return {3, 4};
}

class KernelGetNonConstTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelGetNonConstTest>([=]() {
            {
                dpl::tuple<int> t(3);
                ret_access[0] = (dpl::get<0>(t) == 3);
                dpl::get<0>(t) = 2;
                ret_access[0] &= (dpl::get<0>(t) == 2);
            }

            { // get on an rvalue tuple
                static_assert(dpl::get<0>(dpl::make_tuple(0.0f, 1, 2.0, 3L)) == 0);
                static_assert(dpl::get<1>(dpl::make_tuple(0.0f, 1, 2.0, 3L)) == 1);
                static_assert(dpl::get<2>(dpl::make_tuple(0.0f, 1, 2.0, 3L)) == 2);
                static_assert(dpl::get<3>(dpl::make_tuple(0.0f, 1, 2.0, 3L)) == 3);
                static_assert(S().k == 1);
                static_assert(dpl::get<1>(getP()) == 4);
            }
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Wrong result of dpl::get(dpl::tuple) check");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    kernel_test();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
