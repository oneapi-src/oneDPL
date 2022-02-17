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

#if TEST_DPCPP_BACKEND_PRESENT
#    include <CL/sycl.hpp>
#endif // TEST_DPCPP_BACKEND_PRESENT

#include <vector>

#if TEST_DPCPP_BACKEND_PRESENT
auto
get_sycl_buffer(std::vector<bool>& container)
{
    return cl::sycl::buffer<bool, 1>(container.cbegin(), container.cend());
}
#endif

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue q;
    std::vector<bool> v{ true, false, false };

    auto buf = get_sycl_buffer(v);

    q.submit(
        [&](sycl::handler& cgh)
        {
            auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
            cgh.single_task<class KernelTest>(
                [=]()
                {
                    acc[2] = acc[0] || acc[1];
                });
        }).wait();

#endif

    return 0;
}