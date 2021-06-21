//===-- xpu_equal.pass.cpp ------------------------------------------------===//
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

#include <oneapi/dpl/algorithm>

#include "support/test_iterators.h"

#include <cassert>
#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

using oneapi::dpl::equal;

void
kernel_test1(sycl::queue& deviceQueue)
{
    sycl::cl_bool ret = false;
    sycl::range<1> item1{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                const int s = 6;
                int ia[] = {0, 1, 2, 3, 4, 5};
                int ib[s] = {0, 1, 2, 5, 4, 5};

                ret_acc[0] = (equal(input_iterator<const int*>(ia), input_iterator<const int*>(ia + s),
                                    input_iterator<const int*>(ia)));
                ret_acc[0] &= (!equal(input_iterator<const int*>(ia), input_iterator<const int*>(ia + s),
                                      input_iterator<const int*>(ib)));
                ret_acc[0] &=
                    (!equal(random_access_iterator<const int*>(ia), random_access_iterator<const int*>(ia + s),
                            random_access_iterator<const int*>(ib)));
            });
        });
    }
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue;
    kernel_test1(deviceQueue);
    return 0;
}
