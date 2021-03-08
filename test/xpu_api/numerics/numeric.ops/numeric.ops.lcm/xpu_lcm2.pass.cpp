// -*- C++ -*-
//===-- xpu_lcm2.pass.cpp --------------------------------------------===//
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

#include <iostream>

#include <CL/sycl.hpp>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/type_traits>

class KernelTest;

void
test()
{
    cl::sycl::queue deviceQueue;
    {
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<KernelTest>([=]() {
                static_assert(oneapi::dpl::lcm(21, 6) == 42, "");
                static_assert(oneapi::dpl::lcm(41, 0) == 0, "LCD with zero is zero");
                static_assert(oneapi::dpl::lcm(0, 7) == 0, "LCD with zero is zero");
                static_assert(oneapi::dpl::lcm(0, 0) == 0, "no division by zero");

                static_assert(oneapi::dpl::lcm(1u, 2) == 2, "unsigned and signed");
                static_assert(oneapi::dpl::lcm(3, 4u) == 12, "signed and unsigned");
                static_assert(oneapi::dpl::lcm(5u, 6u) == 30, "unsigned and unsigned");

                static_assert(oneapi::dpl::is_same_v<decltype(oneapi::dpl::lcm(1l, 1)), long>);
                static_assert(oneapi::dpl::is_same_v<decltype(oneapi::dpl::lcm(1ul, 1ull)), unsigned long long>);
                // PR libstdc++/92978
                static_assert(oneapi::dpl::lcm(-42, 21U) == 42U);
            });
        });
    }
}

int
main()
{
    test();
    std::cout << "done" << std::endl;

    return 0;
}
