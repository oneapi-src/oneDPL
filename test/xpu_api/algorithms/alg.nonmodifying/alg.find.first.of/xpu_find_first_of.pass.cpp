//===-- xpu_find_first_of.pass.cpp ----------------------------------------===//
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

using oneapi::dpl::find_first_of;

void
kernel_test(sycl::queue& deviceQueue)
{
    sycl::cl_bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<sycl::cl_bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl_write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                int ia[] = {0, 1, 2, 3, 0, 1, 2, 3};
                const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                int ib[] = {1, 3, 5, 7};
                const unsigned sb = sizeof(ib) / sizeof(ib[0]);
                ret_acc[0] &= find_first_of(input_iterator<const int*>(ia), input_iterator<const int*>(ia + sa),
                                            forward_iterator<const int*>(ib), forward_iterator<const int*>(ib + sb)) ==
                              input_iterator<const int*>(ia + 1);
                int ic[] = {7};
                ret_acc[0] &= find_first_of(input_iterator<const int*>(ia), input_iterator<const int*>(ia + sa),
                                            forward_iterator<const int*>(ic), forward_iterator<const int*>(ic + 1)) ==
                              input_iterator<const int*>(ia + sa);
                ret_acc[0] &= find_first_of(input_iterator<const int*>(ia), input_iterator<const int*>(ia + sa),
                                            forward_iterator<const int*>(ic),
                                            forward_iterator<const int*>(ic)) == input_iterator<const int*>(ia + sa);
                ret_acc[0] &= find_first_of(input_iterator<const int*>(ia), input_iterator<const int*>(ia),
                                            forward_iterator<const int*>(ic),
                                            forward_iterator<const int*>(ic + 1)) == input_iterator<const int*>(ia);
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
