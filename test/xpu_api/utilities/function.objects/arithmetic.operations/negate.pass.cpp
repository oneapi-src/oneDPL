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

#include <oneapi/dpl/functional>

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
class KernelNegateTest;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelNegateTest>([=]() {
            typedef dpl::negate<int> Fint;
            const Fint fi = Fint();
            ret_access[0] = (fi(36) == -36);

            typedef dpl::negate<long> Flong;
            const Flong fl = Flong();
            ret_access[0] &= (fl(-36L) == 36);

            typedef dpl::negate<float> Ffloat;
            const Ffloat ff = Ffloat();
            ret_access[0] &= (ff(36.0f) == -36.0f);
        });
    });

    auto ret_access_host = buffer1.get_access<sycl::access::mode::read>();
    EXPECT_TRUE(ret_access_host[0], "Error in work with dpl::negate");
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
