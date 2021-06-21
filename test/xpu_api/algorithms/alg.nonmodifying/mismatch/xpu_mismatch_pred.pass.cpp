//===-- xpu_mismatch_pred.pass.cpp ----------------------------------------===//
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
#include <oneapi/dpl/utility>

#include "support/test_iterators.h"

#include <cassert>
#include <CL/sycl.hpp>

constexpr sycl::access::mode sycl_read = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write = sycl::access::mode::write;

using oneapi::dpl::mismatch;
using oneapi::dpl::pair;

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
                int ia[] = {0, 1, 2, 2, 0, 1, 2, 3};
                const unsigned sa = sizeof(ia) / sizeof(ia[0]);
                int ib[] = {0, 1, 2, 3, 0, 1, 2, 3};
                const unsigned sb = sizeof(ib) / sizeof(ib[0]);

                typedef input_iterator<const int*> II;
                typedef random_access_iterator<const int*> RAI;

                ret_acc[0] &= (mismatch(II(ia), II(ia + sa), II(ib)) == (pair<II, II>(II(ia + 3), II(ib + 3))));

                ret_acc[0] &= (mismatch(RAI(ia), RAI(ia + sa), RAI(ib)) == (pair<RAI, RAI>(RAI(ia + 3), RAI(ib + 3))));

                ret_acc[0] &=
                    (mismatch(II(ia), II(ia + sa), II(ib), II(ib + sb)) == (pair<II, II>(II(ia + 3), II(ib + 3))));

                ret_acc[0] &= (mismatch(RAI(ia), RAI(ia + sa), RAI(ib), RAI(ib + sb)) ==
                               (pair<RAI, RAI>(RAI(ia + 3), RAI(ib + 3))));

                ret_acc[0] &=
                    (mismatch(II(ia), II(ia + sa), II(ib), II(ib + 2)) == (pair<II, II>(II(ia + 2), II(ib + 2))));
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
