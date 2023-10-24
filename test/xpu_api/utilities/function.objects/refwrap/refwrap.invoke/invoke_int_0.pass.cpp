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

// <functional>

// reference_wrapper

// template <class... ArgTypes>
//   requires Callable<T, ArgTypes&&...>
//   Callable<T, ArgTypes&&...>::result_type
//   operator()(ArgTypes&&... args) const;


#include "support/test_config.h"

#include <oneapi/dpl/functional>

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
// 0 args, return int

int count = 0;

struct A_int_0
{
    int
    operator()()
    {
        return 4;
    }
};

bool
test_int_0()
{
    {
        A_int_0 a0;
        std::reference_wrapper<A_int_0> r1(a0);
        return (r1() == 4);
    }
}

class KernelInvokeInt0Test;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelInvokeInt0Test>([=]() { ret_access[0] = test_int_0(); });
    });

    auto ret_access_host = buffer1.get_access<sycl::access::mode::read>();
    EXPECT_TRUE(ret_access_host[0], "Error in work with invoke (int)");
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
