//===-- xpu_find_if_not.pass.cpp ------------------------------------------===//
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

using oneapi::dpl::find_if_not;

struct ne
{
    ne(int val) : v(val) {}
    bool
    operator()(int v2) const
    {
        return v != v2;
    }
    int v;
};

void
kernel_test(sycl::queue& deviceQueue)
{
    sycl::cl_bool ret = false;
    sycl::range<1> item1{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                int ia[] = {0, 1, 2, 3, 4, 5};
                const unsigned s = sizeof(ia) / sizeof(ia[0]);
                input_iterator<const int*> r =
                    find_if_not(input_iterator<const int*>(ia), input_iterator<const int*>(ia + s), ne(3));
                ret_acc[0] &= (*r == 3);
                r = find_if_not(input_iterator<const int*>(ia), input_iterator<const int*>(ia + s), ne(10));
                ret_acc[0] &= (r == input_iterator<const int*>(ia + s));
            });
        });
    }
}

int
main()
{
    sycl::queue deviceQueue;
    kernel_test(deviceQueue);
    return 0;
}
