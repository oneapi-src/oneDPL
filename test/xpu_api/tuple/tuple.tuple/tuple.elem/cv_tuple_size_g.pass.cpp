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
bool
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = true;
    sycl::range<1> numOfItems{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest>([=]() {
                ret_access[0] &= (dpl::tuple_size<const dpl::tuple<>>::value == 0);
                ret_access[0] &= (dpl::tuple_size<volatile dpl::tuple<int>>::value == 1);
                ret_access[0] &= (dpl::tuple_size<const volatile dpl::tuple<void>>::value == 1);

                typedef dpl::tuple<int, const int&, void> test_tuple1;
                ret_access[0] &= (dpl::tuple_size<const test_tuple1>::value == 3);
                ret_access[0] &= (dpl::tuple_size<const volatile test_tuple1>::value == 3);
                ret_access[0] &= (dpl::tuple_size<volatile dpl::tuple<dpl::tuple<void>>>::value == 1);
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
    auto ret = kernel_test();
    EXPECT_TRUE(ret, "Wrong result of dpl::tuple_size check in kernel_test");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
