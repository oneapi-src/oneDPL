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

// <array>
// template <class T, size_t N> void swap(array<T,N>& x, array<T,N>& y);

#include "support/test_config.h"

#include <oneapi/dpl/array>

#include "support/utils.h"

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT && !TEST_XPU_ARRAY_SWAP_BROKEN
    bool ret = true;
    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{1});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto ret_acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                {
                    typedef float T;
                    typedef dpl::array<T, 3> C;
                    C c1 = {1.f, 2.f, 3.5f};
                    C c2 = {4.f, 5.f, 6.5f};
                    swap(c1, c2);
                    ret_acc[0] &= (c1.size() == 3);
                    ret_acc[0] &= (c1[0] == 4.f);
                    ret_acc[0] &= (c1[1] == 5.f);
                    ret_acc[0] &= (c1[2] == 6.5f);
                    ret_acc[0] &= (c2.size() == 3);
                    ret_acc[0] &= (c2[0] == 1.f);
                    ret_acc[0] &= (c2[1] == 2.f);
                    ret_acc[0] &= (c2[2] == 3.5f);
                }
                {
                    typedef float T;
                    typedef dpl::array<T, 0> C;
                    C c1 = {};
                    C c2 = {};
                    swap(c1, c2);
                    ret_acc[0] &= (c1.size() == 0);
                    ret_acc[0] &= (c2.size() == 0);
                }
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with dpl::swap(dpl::array, dpl::array)");
#endif // TEST_DPCPP_BACKEND_PRESENT &&  && !TEST_XPU_ARRAY_SWAP_BROKEN

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && !TEST_XPU_ARRAY_SWAP_BROKEN);
}
