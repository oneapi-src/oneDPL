//===-- xpu_find_if.pass.cpp ----------------------------------------------===//
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

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

using oneapi::dpl::find_if;

struct eq
{
    eq(int val) : v(val) {}
    bool
    operator()(int v2) const
    {
        return v == v2;
    }
    int v;
};

void
kernel_test(cl::sycl::queue& deviceQueue)
{
    cl::sycl::cl_bool ret = true;
    cl::sycl::range<1> item1{1};
    {
        cl::sycl::buffer<cl::sycl::cl_bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                int ia[] = {0, 1, 2, 3, 4, 5};
                const unsigned s = sizeof(ia) / sizeof(ia[0]);
                input_iterator<const int*> r =
                    find_if(input_iterator<const int*>(ia), input_iterator<const int*>(ia + s), eq(3));
                ret_acc[0] &= (*r == 3);
                r = find_if(input_iterator<const int*>(ia), input_iterator<const int*>(ia + s), eq(10));
                ret_acc[0] &= (r == input_iterator<const int*>(ia + s));
            });
        });
    }
    assert(ret);
}

int
main()
{
    sycl::queue deviceQueue;
    kernel_test(deviceQueue);
    return 0;
}
