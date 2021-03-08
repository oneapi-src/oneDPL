// -*- C++ -*-
//===-- xpu_gcd2.pass.cpp --------------------------------------------===//
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

using oneapi::dpl::gcd;
using oneapi::dpl::is_same_v;

class KernelTest;

void
test()
{
    cl::sycl::queue deviceQueue;
    {
        deviceQueue.submit([&](cl::sycl::handler& cgh) {
            cgh.single_task<KernelTest>([=]() {
                static_assert(gcd(1071, 462) == 21, "");
                static_assert(gcd(2000, 20) == 20, "");
                static_assert(gcd(2011, 17) == 1, "GCD of two primes is 1");
                static_assert(gcd(200, 200) == 200, "GCD of equal numbers is that number");
                static_assert(gcd(0, 13) == 13, "GCD of any number and 0 is that number");
                static_assert(gcd(29, 0) == 29, "GCD of any number and 0 is that number");
                static_assert(gcd(0, 0) == 0, "Zarro Boogs found");

                static_assert(gcd(1u, 2) == 1, "unsigned and signed");
                static_assert(gcd(9, 6u) == 3, "unsigned and signed");
                static_assert(gcd(3, 4u) == 1, "signed and unsigned");
                static_assert(gcd(32u, 24) == 8, "signed and unsigned");
                static_assert(gcd(1u, -2) == 1, "unsigned and negative");
                static_assert(gcd(-3, 4u) == 1, "negative and unsigned");
                static_assert(gcd(5u, 6u) == 1, "unsigned and unsigned");
                static_assert(gcd(54u, 36u) == 18, "unsigned and unsigned");
                static_assert(gcd(-5, -6) == 1, "negative and negative");
                static_assert(gcd(-50, -60) == 10, "negative and negative");
                static_assert(gcd(-21, 28u) == 7);
                static_assert(gcd(33u, -44) == 11);

                static_assert(is_same_v<decltype(gcd(1l, 1)), long>);
                static_assert(is_same_v<decltype(gcd(1ul, 1ull)), unsigned long long>);

                // PR libstdc++/92978
                static_assert(gcd(-120, 10U) == 10);
                static_assert(gcd(120U, -10) == 10);

                // |INT_MIN| should not be undefined, as long as it fits in the result type.
                static_assert(gcd(INT_MIN, 0LL) == 1LL + INT_MAX);
                static_assert(gcd(0LL, INT_MIN) == 1LL + INT_MAX);
                static_assert(gcd(INT_MIN, 0LL + INT_MIN) == 1LL + INT_MAX);
                static_assert(gcd(INT_MIN, 1LL + INT_MAX) == 1LL + INT_MAX);
                static_assert(gcd(SHRT_MIN, 1U + SHRT_MAX) == 1U + SHRT_MAX);
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
